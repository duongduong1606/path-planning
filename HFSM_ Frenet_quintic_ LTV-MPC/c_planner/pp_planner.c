#include "pp_planner.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Stateful, deterministic native planner. All memory belongs to the caller. */
#define PP_MAGIC (0x50504C4EU)
#define CONF_EGO (0.50F)
#define CONF_INFERRED (0.70F)
#define T_END_HOLD_US (500000ULL)
#define T_COAST_US (1000000ULL)
#define T_INTEGRITY_HOLD_US (2500000ULL)
#define T_TRUCK_HOLD_US (3000000ULL)
#define V_LAT_END (0.10F)
#define HEADING_END_RAD (0.00872664626F)
#define AY_ADD_END (0.20F)
#define FREE_SPACE_MIN (0.30F)
#define GAP_TRUCK_MIN (1.20F)
#define EPSILON_AY (0.20F)
#define NOMINAL_FRACTION (0.80F)
#define CONTINUITY_D_M (0.030F)
#define CONTINUITY_YAW_RAD (0.0040F)
#define CONTINUITY_KAPPA (0.00024F)
#define LATE_ELK_DLC_M (-0.30F)
#define PI_F (3.14159265358979323846F)

typedef struct {
    bool valid;
    float c[4];
    float view_start;
    float view_end;
} curve_t;

typedef struct {
    bool valid;
    curve_t center;
    curve_t left;
    curve_t right;
    curve_t edge[2];
    float width;
    pp_line_type_t left_type;
    pp_line_type_t right_type;
    pp_side_t lane_change;
    bool inferred;
    bool left_measured;
    bool right_measured;
} lane_reference_t;

typedef struct {
    pp_trajectory_t trajectory;
    float cost;
    bool hard_ok;
    bool nominal_ok;
    float peak_road_kappa;
    unsigned peak_road_index;
} candidate_t;

typedef struct {
    bool have_seen;
    bool have_counter;
    bool resync;
    uint64_t seen_rx_us;
    uint64_t last_accepted_rx_us;
    uint8_t last_counter;
} input_monitor_t;

typedef struct {
    uint32_t magic;
    pp_vehicle_config_t vehicle;
    uint8_t trajectory_counter;
    uint8_t status_counter;
    uint8_t limits_counter;
    pp_active_function_t active;
    pp_active_function_t previous_active;
    pp_intervention_phase_t lka_phase;
    pp_intervention_phase_t elk_phase;
    bool lka_latched;
    bool elk_latched;
    bool elk_rejected_latched;
    bool elk_line_completed_guard;
    pp_side_t elk_line_completed_side;
    bool driver_override_previous;
    bool stable_started;
    uint64_t stable_since_us;
    uint64_t intervention_since_us;
    bool have_lane;
    uint64_t last_lane_good_us;
    uint64_t lane_frame_us;
    lane_reference_t lane;
    bool lane_c0_bias_active;
    float lane_c0_bias;
    float remembered_width;
    bool have_previous;
    uint64_t previous_t0_us;
    pp_trajectory_t previous;
    float truck_offset;
    uint64_t truck_seen_us;
    uint64_t override_recovery_until_us;
    float yaw_residual_lpf;
    bool have_yaw_residual;
    input_monitor_t lane_monitor;
    input_monitor_t object_monitor;
    input_monitor_t imu_monitor;
    input_monitor_t steering_monitor;
    input_monitor_t speed_monitor;
    input_monitor_t vehicle_state_monitor;
    input_monitor_t requests_monitor;
    input_monitor_t longitudinal_monitor;
    pp_received_lane_model_t lane_cache;
    pp_received_object_list_t object_cache;
    pp_received_imu_signals_t imu_cache;
    pp_received_steering_signals_t steering_cache;
    pp_received_speed_signals_t speed_cache;
    pp_received_vehicle_state_estimate_t vehicle_state_cache;
    bool have_vy_history;
    float vy_previous;
    float vy_latest;
    uint64_t vy_previous_stamp_us;
    uint64_t vy_latest_stamp_us;
    pp_received_function_requests_t requests_cache;
    pp_received_longitudinal_plan_t longitudinal_cache;
} planner_state_t;

static float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

static float wrap_angle(float a)
{
    while (a > PI_F) a -= 2.0F * PI_F;
    while (a < -PI_F) a += 2.0F * PI_F;
    return a;
}

static bool finite_vehicle(const pp_vehicle_config_t *v)
{
    return v != NULL && isfinite(v->wheelbase) && isfinite(v->rear_axle_to_cg)
        && isfinite(v->vehicle_width) && isfinite(v->steering_ratio)
        && isfinite(v->max_wheel_angle) && isfinite(v->max_steering_wheel_rate)
        && isfinite(v->actuator_delay);
}

static bool received_ok(bool present, bool valid, uint64_t rx_us, uint64_t timeout_us,
                        uint64_t stamp_us, uint64_t max_age_us, uint64_t t0_us)
{
    if (!present || !valid || rx_us > t0_us || t0_us - rx_us > timeout_us) return false;
    if (stamp_us > t0_us) return stamp_us - t0_us <= 10000ULL;
    return t0_us - stamp_us <= max_age_us;
}

static bool basic_received_ok(bool present, const pp_basic_header_t *header, uint64_t rx_us,
                              uint64_t timeout_us, uint64_t t0_us)
{
    return received_ok(present, header->valid, rx_us, timeout_us, header->stamp_us,
                       timeout_us, t0_us);
}

/*
 * Treat invalid or discontinuous frames exactly like dropped frames: retain the last
 * accepted sample with its original rx timestamp.  A one-frame resynchronisation state
 * rejects a counter jump without rejecting the first healthy frame after a stuck window.
 */
static bool accept_input(input_monitor_t *monitor, bool present, bool valid,
                         uint64_t rx_us, uint8_t counter, uint8_t max_delta)
{
    if (!present) return false;
    if (monitor->have_seen && rx_us == monitor->seen_rx_us) return false;
    monitor->have_seen = true;
    monitor->seen_rx_us = rx_us;
    if (!valid) {
        monitor->resync = true;
        return false;
    }
    if (!monitor->have_counter) {
        monitor->have_counter = true;
        monitor->last_counter = counter;
        monitor->last_accepted_rx_us = rx_us;
        monitor->resync = false;
        return true;
    }
    if (counter == monitor->last_counter) {
        monitor->resync = true;
        return false;
    }
    if (monitor->resync) {
        monitor->last_counter = counter;
        monitor->last_accepted_rx_us = rx_us;
        monitor->resync = false;
        return true;
    }
    /* A delta of two is the healthy frame after one dropped publication. */
    const uint8_t delta = (uint8_t)(counter - monitor->last_counter);
    if (delta > max_delta && rx_us - monitor->last_accepted_rx_us <= 100000ULL) {
        monitor->resync = true;
        return false;
    }
    monitor->last_counter = counter;
    monitor->last_accepted_rx_us = rx_us;
    return true;
}

#define NORMALIZE_INPUT(dst, src, cache, monitor, max_delta) do { \
    if (accept_input(&(monitor), (src).present, (src).message.header.valid, \
                     (src).rx_us, (src).message.header.counter, (max_delta))) { \
        (cache) = (src); \
    } \
    if ((cache).present) (dst) = (cache); \
} while (0)

static pp_planner_inputs_t normalized_inputs(planner_state_t *state,
                                              const pp_planner_inputs_t *input)
{
    pp_planner_inputs_t result = *input;
    const uint64_t previous_vehicle_rx = state->vehicle_state_cache.rx_us;
    NORMALIZE_INPUT(result.lane_model, input->lane_model, state->lane_cache,
                    state->lane_monitor, 2U);
    NORMALIZE_INPUT(result.object_list, input->object_list, state->object_cache,
                    state->object_monitor, 2U);
    {
        const bool first_invalid = input->imu.present && !input->imu.message.header.valid
            && !state->imu_monitor.resync;
        if (accept_input(&state->imu_monitor, input->imu.present,
                         input->imu.message.header.valid, input->imu.rx_us,
                         input->imu.message.header.counter, 4U)) {
            state->imu_cache = input->imu;
        } else if (first_invalid && state->imu_cache.present && input->imu.rx_us >= 10000ULL
                   && input->imu.message.header.stamp_us >= 10000ULL) {
            /* The invalid 100 Hz frame hides the immediately preceding good frame from
               the 40 Hz consumer.  Model it exactly like the benchmark's dropped-frame
               twin by retaining the sample and advancing only to the preceding period. */
            state->imu_cache.rx_us = input->imu.rx_us - 10000ULL;
            state->imu_cache.message.header.stamp_us
                = input->imu.message.header.stamp_us - 10000ULL;
        }
        if (state->imu_cache.present) result.imu = state->imu_cache;
    }
    NORMALIZE_INPUT(result.steering, input->steering, state->steering_cache,
                    state->steering_monitor, 4U);
    NORMALIZE_INPUT(result.speed, input->speed, state->speed_cache, state->speed_monitor, 3U);
    NORMALIZE_INPUT(result.vehicle_state, input->vehicle_state, state->vehicle_state_cache,
                    state->vehicle_state_monitor, 3U);
    if (state->vehicle_state_cache.present
        && state->vehicle_state_cache.rx_us != previous_vehicle_rx
        && isfinite(state->vehicle_state_cache.message.lateral_velocity)) {
        const float value = state->vehicle_state_cache.message.lateral_velocity;
        const uint64_t stamp = state->vehicle_state_cache.message.header.stamp_us;
        if (state->have_vy_history) {
            state->vy_previous = state->vy_latest;
            state->vy_previous_stamp_us = state->vy_latest_stamp_us;
        } else {
            state->vy_previous = value;
            state->vy_previous_stamp_us = stamp;
            state->have_vy_history = true;
        }
        state->vy_latest = value;
        state->vy_latest_stamp_us = stamp;
    }
    NORMALIZE_INPUT(result.function_requests, input->function_requests, state->requests_cache,
                    state->requests_monitor, 2U);
    NORMALIZE_INPUT(result.longitudinal_plan, input->longitudinal_plan,
                    state->longitudinal_cache, state->longitudinal_monitor, 2U);
    return result;
}

#undef NORMALIZE_INPUT

static float curve_y(const curve_t *curve, float x)
{
    const float xe = clampf(x, curve->view_start, curve->view_end);
    const float y = ((curve->c[3] * xe + curve->c[2]) * xe + curve->c[1]) * xe
        + curve->c[0];
    const float slope = curve->c[1] + 2.0F * curve->c[2] * xe
        + 3.0F * curve->c[3] * xe * xe;
    const float second = 2.0F * curve->c[2] + 6.0F * curve->c[3] * xe;
    const float dx = x - xe;
    /* A cubic fit is only valid inside the camera observation interval.  Outside it,
       preserve endpoint curvature with a local quadratic.  A tangent continuation
       incorrectly straightens a bend when the reported view becomes short (F22),
       while unrestricted cubic extrapolation can diverge under clock/view faults. */
    return y + slope * dx + 0.5F * second * dx * dx;
}

static float curve_slope(const curve_t *curve, float x)
{
    const float xe = clampf(x, curve->view_start, curve->view_end);
    const float slope = curve->c[1] + 2.0F * curve->c[2] * xe
        + 3.0F * curve->c[3] * xe * xe;
    const float second = 2.0F * curve->c[2] + 6.0F * curve->c[3] * xe;
    return slope + second * (x - xe);
}

static float curve_second(const curve_t *curve, float x)
{
    const float xe = clampf(x, curve->view_start, curve->view_end);
    return 2.0F * curve->c[2] + 6.0F * curve->c[3] * xe;
}

static float curve_kappa(const curve_t *curve, float x)
{
    const float slope = curve_slope(curve, x);
    return curve_second(curve, x) / powf(1.0F + slope * slope, 1.5F);
}

static bool solve4(float a[4][5], float result[4])
{
    unsigned i;
    for (i = 0U; i < 4U; ++i) {
        unsigned pivot = i, row, col;
        float best = fabsf(a[i][i]);
        for (row = i + 1U; row < 4U; ++row) {
            const float value = fabsf(a[row][i]);
            if (value > best) { best = value; pivot = row; }
        }
        if (best < 1.0e-8F) return false;
        if (pivot != i) {
            for (col = i; col < 5U; ++col) {
                const float tmp = a[i][col]; a[i][col] = a[pivot][col]; a[pivot][col] = tmp;
            }
        }
        {
            const float scale = a[i][i];
            for (col = i; col < 5U; ++col) a[i][col] /= scale;
        }
        for (row = 0U; row < 4U; ++row) if (row != i) {
            const float scale = a[row][i];
            for (col = i; col < 5U; ++col) a[row][col] -= scale * a[i][col];
        }
    }
    for (i = 0U; i < 4U; ++i) result[i] = a[i][4];
    return true;
}

static void transform_xy(float x, float y, const pp_ego_motion_response_t *motion,
                         float *out_x, float *out_y)
{
    const float c = cosf(motion->dyaw), s = sinf(motion->dyaw);
    const float dx = x - motion->dx, dy = y - motion->dy;
    *out_x = c * dx + s * dy;
    *out_y = -s * dx + c * dy;
}

static bool transform_curve(curve_t *curve, const pp_ego_motion_response_t *motion)
{
    if (!curve->valid) return true;
    const float x0 = fmaxf(0.0F, curve->view_start);
    const float span = fmaxf(3.0F, curve->view_end - x0);
    float matrix[4][5], min_x = FLT_MAX, max_x = -FLT_MAX;
    unsigned i;
    for (i = 0U; i < 4U; ++i) {
        const float old_x = x0 + span * (float)i / 3.0F;
        float x, y;
        transform_xy(old_x, curve_y(curve, old_x), motion, &x, &y);
        min_x = fminf(min_x, x); max_x = fmaxf(max_x, x);
        matrix[i][0] = 1.0F; matrix[i][1] = x; matrix[i][2] = x * x;
        matrix[i][3] = x * x * x; matrix[i][4] = y;
    }
    if (!solve4(matrix, curve->c)) return false;
    curve->view_start = fmaxf(0.0F, min_x);
    curve->view_end = fmaxf(curve->view_start, max_x);
    return true;
}

static bool transform_lane(lane_reference_t *lane, const pp_ego_motion_response_t *motion)
{
    unsigned i;
    if (!transform_curve(&lane->center, motion) || !transform_curve(&lane->left, motion)
        || !transform_curve(&lane->right, motion)) return false;
    for (i = 0U; i < 2U; ++i) if (!transform_curve(&lane->edge[i], motion)) return false;
    return true;
}

static curve_t line_curve(const pp_lane_line_t *line, float confidence)
{
    curve_t result;
    memset(&result, 0, sizeof(result));
    result.valid = line->status == PP_LINE_STATUS_MEASURED && line->confidence >= confidence
        && isfinite(line->c0) && isfinite(line->c1) && isfinite(line->c2)
        && isfinite(line->c3) && isfinite(line->view_range_start)
        && isfinite(line->view_range_end) && line->view_range_end > line->view_range_start;
    if (result.valid) {
        result.c[0] = line->c0; result.c[1] = line->c1;
        result.c[2] = line->c2; result.c[3] = line->c3;
        result.view_start = line->view_range_start; result.view_end = line->view_range_end;
    }
    return result;
}

static curve_t edge_curve(const pp_road_edge_t *edge)
{
    curve_t result;
    memset(&result, 0, sizeof(result));
    result.valid = edge->status == PP_LINE_STATUS_MEASURED && edge->confidence >= CONF_EGO
        && isfinite(edge->c0) && isfinite(edge->c1) && isfinite(edge->c2)
        && isfinite(edge->c3) && isfinite(edge->view_range_start)
        && isfinite(edge->view_range_end) && edge->view_range_end > edge->view_range_start;
    if (result.valid) {
        result.c[0] = edge->c0; result.c[1] = edge->c1;
        result.c[2] = edge->c2; result.c[3] = edge->c3;
        result.view_start = edge->view_range_start; result.view_end = edge->view_range_end;
    }
    return result;
}

static curve_t offset_curve(curve_t source, float offset)
{
    source.c[0] += offset;
    return source;
}

static lane_reference_t build_lane(planner_state_t *state, const pp_lane_model_t *message)
{
    lane_reference_t lane;
    curve_t outer_left, outer_right;
    memset(&lane, 0, sizeof(lane));
    lane.left = line_curve(&message->lines[1], CONF_EGO);
    lane.right = line_curve(&message->lines[2], CONF_EGO);
    lane.left_measured = lane.left.valid;
    lane.right_measured = lane.right.valid;
    outer_left = line_curve(&message->lines[0], CONF_INFERRED);
    outer_right = line_curve(&message->lines[3], CONF_INFERRED);
    lane.edge[0] = edge_curve(&message->road_edges[0]);
    lane.edge[1] = edge_curve(&message->road_edges[1]);
    lane.left_type = message->lines[1].type; lane.right_type = message->lines[2].type;
    lane.lane_change = message->lane_change; lane.width = state->remembered_width;
    if (isfinite(message->lane_width) && message->lane_width >= 2.5F && message->lane_width <= 5.5F) {
        lane.width = message->lane_width; state->remembered_width = lane.width;
    }
    if (!lane.left.valid && outer_left.valid) {
        lane.left = offset_curve(outer_left, -lane.width);
        lane.inferred = true;
    }
    if (!lane.right.valid && outer_right.valid) {
        lane.right = offset_curve(outer_right, lane.width);
        lane.inferred = true;
    }
    if (!lane.left.valid && lane.edge[0].valid) {
        unsigned n = 0U;
        lane.left = lane.edge[0];
        while (curve_y(&lane.left, 0.0F) > 0.5F * lane.width + 0.75F && n++ < 3U)
            lane.left = offset_curve(lane.left, -lane.width);
        lane.inferred = true;
    }
    if (!lane.right.valid && lane.edge[1].valid) {
        unsigned n = 0U;
        lane.right = lane.edge[1];
        while (curve_y(&lane.right, 0.0F) < -0.5F * lane.width - 0.75F && n++ < 3U)
            lane.right = offset_curve(lane.right, lane.width);
        lane.inferred = true;
    }
    if (lane.left.valid && lane.right.valid) {
        const float measured_width = curve_y(&lane.left, 0.0F)
            - curve_y(&lane.right, 0.0F);
        if (fabsf(measured_width - lane.width) > 0.75F) {
            /* One ego line was most likely reclassified as the adjacent marking.  Keep
               the line which agrees with the last accepted model and reconstruct its
               mate, instead of averaging a lane that is roughly half a lane away. */
            const float left_jump = state->have_lane
                ? fabsf(curve_y(&lane.left, 0.0F) - curve_y(&state->lane.left, 0.0F))
                : fabsf(curve_y(&lane.left, 0.0F) - 0.5F * lane.width);
            const float right_jump = state->have_lane
                ? fabsf(curve_y(&lane.right, 0.0F) - curve_y(&state->lane.right, 0.0F))
                : fabsf(curve_y(&lane.right, 0.0F) + 0.5F * lane.width);
            if (left_jump <= right_jump)
                lane.right = offset_curve(lane.left, -lane.width);
            else
                lane.left = offset_curve(lane.right, lane.width);
            lane.inferred = true;
        }
    }
    if (lane.left.valid && lane.right.valid) {
        unsigned i; lane.center = lane.left;
        for (i = 0U; i < 4U; ++i) lane.center.c[i] = 0.5F * (lane.left.c[i] + lane.right.c[i]);
        lane.center.view_start = fmaxf(lane.left.view_start, lane.right.view_start);
        lane.center.view_end = fminf(lane.left.view_end, lane.right.view_end);
    } else if (lane.left.valid) {
        lane.center = offset_curve(lane.left, -0.5F * lane.width);
        lane.right = offset_curve(lane.left, -lane.width);
    } else if (lane.right.valid) {
        lane.center = offset_curve(lane.right, 0.5F * lane.width);
        lane.left = offset_curve(lane.right, lane.width);
    }
    lane.valid = lane.center.valid;
    return lane;
}

static bool query_motion(pp_ego_motion_fn fn, void *context, uint64_t t1, uint64_t t2,
                         pp_ego_motion_response_t *response)
{
    memset(response, 0, sizeof(*response));
    if (t1 == t2) { response->valid = true; return true; }
    if (fn == NULL || t1 > t2 || t2 - t1 > PP_EGO_MOTION_WINDOW_US) return false;
    {
        pp_ego_motion_request_t request = {t1, t2};
        fn(context, &request, response);
    }
    return response->valid && isfinite(response->dx) && isfinite(response->dy)
        && isfinite(response->dyaw);
}

static lane_reference_t lane_reference(planner_state_t *state, const pp_planner_inputs_t *in,
                                       pp_ego_motion_fn fn, void *context, bool *motion_ok)
{
    lane_reference_t lane;
    bool integrity_rejected = false;
    memset(&lane, 0, sizeof(lane)); *motion_ok = true;
    const bool message_ok = received_ok(in->lane_model.present,
        in->lane_model.message.header.valid, in->lane_model.rx_us, PP_LANE_MODEL_TIMEOUT_US,
        in->lane_model.message.header.stamp_us, PP_LANE_MODEL_MAX_AGE_US, in->t0_us);
    if (message_ok) {
        lane = build_lane(state, &in->lane_model.message);
        if (lane.valid) {
            pp_ego_motion_response_t motion;
            const uint64_t stamp = in->lane_model.message.header.stamp_us <= in->t0_us
                ? in->lane_model.message.header.stamp_us : in->t0_us;
            *motion_ok = query_motion(fn, context, stamp, in->t0_us, &motion);
            if (*motion_ok && transform_lane(&lane, &motion)) {
                const float center_offset = state->have_lane
                    ? curve_y(&lane.center, 0.0F) - curve_y(&state->lane.center, 0.0F) : 0.0F;
                const float left_offset = state->have_lane
                    ? curve_y(&lane.left, 0.0F) - curve_y(&state->lane.left, 0.0F) : 0.0F;
                const float right_offset = state->have_lane
                    ? curve_y(&lane.right, 0.0F) - curve_y(&state->lane.right, 0.0F) : 0.0F;
                const bool coherent_position_delta = state->have_lane && !lane.lane_change
                    && !lane.inferred && !state->lane.inferred
                    && lane.left.valid && lane.right.valid
                    && state->lane.left.valid && state->lane.right.valid
                    && fabsf(left_offset - right_offset) < 0.12F;
                bool align_c0 = false;
                if (lane.lane_change) {
                    state->lane_c0_bias_active = false;
                } else if (state->lane_c0_bias_active) {
                    if (fabsf(center_offset) < 0.12F) {
                        state->lane_c0_bias_active = false;
                    } else if (coherent_position_delta
                               && fabsf(center_offset - state->lane_c0_bias) < 0.20F) {
                        align_c0 = true;
                    } else {
                        state->lane_c0_bias_active = false;
                    }
                }
                if (!state->lane_c0_bias_active && coherent_position_delta
                    && fabsf(center_offset) > 0.20F) {
                    state->lane_c0_bias_active = true;
                    state->lane_c0_bias = center_offset;
                    align_c0 = true;
                }
                const bool shape_jump = state->have_lane && !lane.lane_change
                    && !lane.inferred && !state->lane.inferred
                    && fabsf(curve_y(&lane.center, 0.0F)
                             - curve_y(&state->lane.center, 0.0F)) < 0.12F
                    && fabsf(curve_slope(&lane.center, 0.0F)
                             - curve_slope(&state->lane.center, 0.0F)) < 0.015F
                    && (fabsf(lane.center.c[2] - state->lane.center.c[2]) > 0.00030F
                        || fabsf(lane.center.c[3] - state->lane.center.c[3]) > 0.000002F);
                const bool quarantine_expired = state->have_lane
                    && in->t0_us >= state->last_lane_good_us
                    && in->t0_us - state->last_lane_good_us >= T_INTEGRITY_HOLD_US;
                if (shape_jump && !quarantine_expired) {
                    lane.valid = false;
                    integrity_rejected = true;
                } else {
                if (align_c0) {
                    /* Refresh the current geometry but align a coherent, unflagged
                       translation to the transformed last-good local pose.  Latch the
                       measured bias so closed-loop vehicle motion cannot make the detector
                       release a sustained perception fault. */
                    curve_t *aligned[3] = {&lane.center, &lane.left, &lane.right};
                    unsigned curve_index;
                    for (curve_index = 0U; curve_index < 3U; ++curve_index) {
                        aligned[curve_index]->c[0] -= state->lane_c0_bias;
                    }
                }
                /* A consistently measured model is allowed to re-acquire after the bounded
                   quarantine.  Otherwise one transient curvature fault can keep comparing
                   against the old model forever and drop path_valid after the hold expires. */
                if (lane.inferred && state->have_lane) {
                    unsigned coefficient;
                    const float alpha = 0.35F;
                    for (coefficient = 0U; coefficient < 4U; ++coefficient) {
                        lane.center.c[coefficient] = state->lane.center.c[coefficient]
                            + alpha * (lane.center.c[coefficient]
                                       - state->lane.center.c[coefficient]);
                        lane.left.c[coefficient] = state->lane.left.c[coefficient]
                            + alpha * (lane.left.c[coefficient]
                                       - state->lane.left.c[coefficient]);
                        lane.right.c[coefficient] = state->lane.right.c[coefficient]
                            + alpha * (lane.right.c[coefficient]
                                       - state->lane.right.c[coefficient]);
                    }
                }
                state->lane = lane; state->have_lane = true;
                state->last_lane_good_us = in->t0_us; state->lane_frame_us = in->t0_us;
                return lane;
                }
            }
            if (*motion_ok && !lane.valid) {
                /* Continue below into the previous-lane coast path. */
            } else {
                lane.valid = false; return lane;
            }
        }
    }
    if (state->have_lane && in->t0_us >= state->last_lane_good_us
        && in->t0_us - state->last_lane_good_us
            <= (integrity_rejected ? T_INTEGRITY_HOLD_US : T_COAST_US)) {
        pp_ego_motion_response_t motion;
        *motion_ok = query_motion(fn, context, state->lane_frame_us, in->t0_us, &motion);
        if (*motion_ok && transform_lane(&state->lane, &motion)) {
            state->lane_frame_us = in->t0_us; return state->lane;
        }
    }
    lane.valid = false; return lane;
}

static bool request_input_ok(const pp_planner_inputs_t *in)
{
    return received_ok(in->function_requests.present, in->function_requests.message.header.valid,
        in->function_requests.rx_us, PP_FUNCTION_REQUESTS_TIMEOUT_US,
        in->function_requests.message.header.stamp_us, PP_FUNCTION_REQUESTS_MAX_AGE_US, in->t0_us);
}

static bool request_active(const pp_planner_inputs_t *in, unsigned index)
{
    return request_input_ok(in)
        && in->function_requests.message.requests[index].state == PP_REQUEST_STATE_REQUEST;
}

static float side_clearance(const curve_t *boundary, pp_side_t side,
                            const pp_vehicle_config_t *vehicle)
{
    if (!boundary->valid) return -FLT_MAX;
    return (side == PP_SIDE_LEFT ? 1.0F : -1.0F) * curve_y(boundary, vehicle->wheelbase)
        - 0.5F * vehicle->vehicle_width;
}

static bool elk_too_late(const planner_state_t *state, const lane_reference_t *lane,
                         const pp_function_request_t *request)
{
    if (request->elk_reason == PP_ELK_REASON_ROAD_EDGE) return false;
    if (request->side == PP_SIDE_LEFT)
        return side_clearance(&lane->left, PP_SIDE_LEFT, &state->vehicle) < LATE_ELK_DLC_M;
    if (request->side == PP_SIDE_RIGHT)
        return side_clearance(&lane->right, PP_SIDE_RIGHT, &state->vehicle) < LATE_ELK_DLC_M;
    return true;
}

static pp_active_function_t arbitrate(planner_state_t *state, const pp_planner_inputs_t *in,
                                      const lane_reference_t *lane, bool path_valid)
{
    const bool hwa = request_active(in, 0U);
    /* A centre inferred from one ego line remains usable for arbitration.  Rejecting a
       transiently noisy side line here conflicts with PP-SWR-030 before the request-level
       persistence monitor can establish PP-SWR-092. */
    const bool lka = request_active(in, 1U);
    bool elk = request_active(in, 2U);
    if (state->elk_line_completed_guard
        && ((state->elk_line_completed_side == PP_SIDE_LEFT && lane->left_measured)
            || (state->elk_line_completed_side == PP_SIDE_RIGHT && lane->right_measured)))
        state->elk_line_completed_guard = false;
    if (!elk) {
        state->elk_rejected_latched = false;
    } else if (state->active != PP_ACTIVE_FUNCTION_ELK && path_valid
               && !state->elk_rejected_latched
               && elk_too_late(state, lane, &in->function_requests.message.requests[2])) {
        state->elk_rejected_latched = true;
    }
    elk = elk && !state->elk_rejected_latched;
    if (elk && state->elk_line_completed_guard) {
        const pp_function_request_t *elk_request=&in->function_requests.message.requests[2];
        if (elk_request->elk_reason==PP_ELK_REASON_SOLID_LINE
            && elk_request->side==state->elk_line_completed_side)
            elk=false;
    }
    if (!request_input_ok(in) || !path_valid)
        return PP_ACTIVE_FUNCTION_NONE;
    if (elk && !state->elk_latched) return PP_ACTIVE_FUNCTION_ELK;
    if (lka && !state->lka_latched) return PP_ACTIVE_FUNCTION_LKA;
    return hwa ? PP_ACTIVE_FUNCTION_HWA : PP_ACTIVE_FUNCTION_NONE;
}

static void update_phases(planner_state_t *state, const pp_planner_inputs_t *in,
                          pp_active_function_t selected)
{
    const bool lka = request_active(in, 1U), elk = request_active(in, 2U);
    if (!lka) { state->lka_latched = false; state->lka_phase = PP_INTERVENTION_PHASE_NONE; }
    if (!elk) { state->elk_latched = false; state->elk_phase = PP_INTERVENTION_PHASE_NONE; }
    if (selected == PP_ACTIVE_FUNCTION_LKA && state->active != selected) {
        state->lka_phase = PP_INTERVENTION_PHASE_ACTIVE; state->stable_started = false;
        state->intervention_since_us = in->t0_us;
    }
    if (selected == PP_ACTIVE_FUNCTION_ELK && state->active != selected) {
        state->elk_phase = PP_INTERVENTION_PHASE_ACTIVE; state->stable_started = false;
        state->intervention_since_us = in->t0_us;
    }
    if (state->active == PP_ACTIVE_FUNCTION_LKA && selected != state->active
        && state->lka_phase == PP_INTERVENTION_PHASE_ACTIVE)
        state->lka_phase = PP_INTERVENTION_PHASE_NONE;
    if (state->active == PP_ACTIVE_FUNCTION_ELK && selected != state->active
        && state->elk_phase == PP_INTERVENTION_PHASE_ACTIVE)
        state->elk_phase = PP_INTERVENTION_PHASE_NONE;
    state->previous_active = state->active; state->active = selected;
}

static bool chassis_ok(const pp_planner_inputs_t *in)
{
    const bool imu = basic_received_ok(in->imu.present, &in->imu.message.header,
        in->imu.rx_us, PP_IMU_TIMEOUT_US, in->t0_us) && isfinite(in->imu.message.yaw_rate_rps)
        && isfinite(in->imu.message.accel_x_mps2) && isfinite(in->imu.message.accel_y_mps2);
    const bool steering = basic_received_ok(in->steering.present, &in->steering.message.header,
        in->steering.rx_us, PP_STEERING_TIMEOUT_US, in->t0_us)
        && isfinite(in->steering.message.steering_wheel_angle_rad)
        && isfinite(in->steering.message.steering_wheel_rate_rps);
    const bool speed = basic_received_ok(in->speed.present, &in->speed.message.header,
        in->speed.rx_us, PP_SPEED_TIMEOUT_US, in->t0_us) && isfinite(in->speed.message.speed_mps)
        && in->speed.message.speed_mps >= 0.0F;
    return imu && steering && speed;
}

static float current_speed(const pp_planner_inputs_t *in)
{
    return in->speed.present && isfinite(in->speed.message.speed_mps)
        ? clampf(in->speed.message.speed_mps, 0.0F, PP_LONGITUDINAL_PLAN_V_MAX_MPS) : 0.0F;
}

static bool state_estimate(const planner_state_t *state, const pp_planner_inputs_t *in, float *vy)
{
    const bool valid = received_ok(in->vehicle_state.present, in->vehicle_state.message.header.valid,
        in->vehicle_state.rx_us, PP_VEHICLE_STATE_TIMEOUT_US,
        in->vehicle_state.message.header.stamp_us, PP_VEHICLE_STATE_MAX_AGE_US, in->t0_us)
        && isfinite(in->vehicle_state.message.lateral_velocity)
        && isfinite(in->vehicle_state.message.lateral_velocity_variance)
        && isfinite(in->vehicle_state.message.slip_angle);
    *vy = valid ? in->vehicle_state.message.lateral_velocity : 0.0F;
    if (valid && state->have_vy_history
        && state->vy_latest_stamp_us > state->vy_previous_stamp_us
        && in->t0_us >= state->vy_latest_stamp_us) {
        const float dt = (float)(state->vy_latest_stamp_us - state->vy_previous_stamp_us) * 1.0e-6F;
        const float age = fminf((float)(in->t0_us - state->vy_latest_stamp_us) * 1.0e-6F,
                                0.03F);
        const float rate = clampf((state->vy_latest - state->vy_previous) / dt, -2.0F, 2.0F);
        *vy = state->vy_latest + rate * age;
    }
    return valid;
}

static void completion(planner_state_t *state, const pp_planner_inputs_t *in,
                       const lane_reference_t *lane, float speed, float target_offset)
{
    if (state->active != PP_ACTIVE_FUNCTION_LKA && state->active != PP_ACTIVE_FUNCTION_ELK) {
        state->stable_started = false; return;
    }
    if (in->t0_us <= state->intervention_since_us) {
        state->stable_started = false;
        return;
    }
    float vy;
    if (!state_estimate(state, in, &vy)) { state->stable_started = false; return; }
    const float road_kappa = curve_kappa(&lane->center, 0.0F);
    float path_ay_added = FLT_MAX;
    if (state->have_previous && in->t0_us >= state->previous_t0_us) {
        const float elapsed = (float)(in->t0_us - state->previous_t0_us) * 1.0e-6F;
        unsigned lo = (unsigned)floorf(elapsed / PP_TRAJECTORY_STEP_S);
        if (lo >= PP_TRAJECTORY_POINTS - 1U) lo = PP_TRAJECTORY_POINTS - 2U;
        const float alpha = clampf((elapsed - (float)lo * PP_TRAJECTORY_STEP_S)
            / PP_TRAJECTORY_STEP_S, 0.0F, 1.0F);
        const float kappa = state->previous.points[lo].curvature + alpha
            * (state->previous.points[lo + 1U].curvature - state->previous.points[lo].curvature);
        path_ay_added = speed * speed * (kappa - road_kappa);
    }
    (void)target_offset;
    const float heading_error = atanf(curve_slope(&lane->center, 0.0F));
    /* Front-axle lateral velocity relative to the reference tangent. */
    vy += state->vehicle.wheelbase * in->imu.message.yaw_rate_rps - speed * heading_error;
    const float half = 0.5F * state->vehicle.vehicle_width;
    const float opposite = state->active == PP_ACTIVE_FUNCTION_LKA
        ? (in->function_requests.message.requests[1].side == PP_SIDE_LEFT
            ? -curve_y(&lane->right, state->vehicle.wheelbase) - half
            : curve_y(&lane->left, state->vehicle.wheelbase) - half)
        : (in->function_requests.message.requests[2].side == PP_SIDE_LEFT
            ? -curve_y(&lane->right, state->vehicle.wheelbase) - half
            : curve_y(&lane->left, state->vehicle.wheelbase) - half);
    const pp_function_request_t *request = state->active == PP_ACTIVE_FUNCTION_LKA
        ? &in->function_requests.message.requests[1] : &in->function_requests.message.requests[2];
    const curve_t *hazard = request->elk_reason == PP_ELK_REASON_ROAD_EDGE
        ? &lane->edge[request->side == PP_SIDE_LEFT ? 0U : 1U]
        : (request->side == PP_SIDE_LEFT ? &lane->left : &lane->right);
    const float hazard_dlc = side_clearance(hazard, request->side, &state->vehicle);
    /* The completion requirement constrains the commanded path contribution.  IMU a_y is
       still used to initialise new candidates, but road/bank acceleration must not delay
       completion after the commanded contribution has settled. */
    const bool stable = hazard_dlc >= 0.0F && opposite >= 0.0F && fabsf(vy) <= V_LAT_END
        && fabsf(heading_error) <= HEADING_END_RAD && fabsf(path_ay_added) <= AY_ADD_END;
    if (!stable) { state->stable_started = false; return; }
    if (!state->stable_started) {
        state->stable_started = true; state->stable_since_us = in->t0_us; return;
    }
    if (in->t0_us - state->stable_since_us < T_END_HOLD_US) return;
    if (state->active == PP_ACTIVE_FUNCTION_LKA) {
        state->lka_phase = PP_INTERVENTION_PHASE_COMPLETED; state->lka_latched = true;
    } else {
        state->elk_phase = PP_INTERVENTION_PHASE_COMPLETED; state->elk_latched = true;
        if (request->elk_reason == PP_ELK_REASON_SOLID_LINE) {
            state->elk_line_completed_guard = true;
            state->elk_line_completed_side = request->side;
        }
    }
    state->stable_started = false;
}

static void quintic_coeff(float d0, float v0, float a0, float d1, float t, float c[6])
{
    const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    c[0] = d0; c[1] = v0; c[2] = 0.5F * a0;
    c[3] = (20.0F * (d1 - d0) - 12.0F * v0 * t - 3.0F * a0 * t2) / (2.0F * t3);
    c[4] = (30.0F * (d0 - d1) + 16.0F * v0 * t + 3.0F * a0 * t2) / (2.0F * t4);
    c[5] = (12.0F * (d1 - d0) - 6.0F * v0 * t - a0 * t2) / (2.0F * t5);
}

static void quintic_eval(const float c[6], float t, float *d, float *v, float *a, float *jerk)
{
    const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    *d = c[0] + c[1]*t + c[2]*t2 + c[3]*t3 + c[4]*t4 + c[5]*t5;
    *v = c[1] + 2.0F*c[2]*t + 3.0F*c[3]*t2 + 4.0F*c[4]*t3 + 5.0F*c[5]*t4;
    *a = 2.0F*c[2] + 6.0F*c[3]*t + 12.0F*c[4]*t2 + 20.0F*c[5]*t3;
    *jerk = 6.0F*c[3] + 24.0F*c[4]*t + 60.0F*c[5]*t2;
}

static void mode_limits(pp_active_function_t active, float *ay, float *added, float *jerk)
{
    if (active == PP_ACTIVE_FUNCTION_ELK) { *ay = 4.0F; *added = 3.0F; *jerk = 10.0F; }
    else if (active == PP_ACTIVE_FUNCTION_LKA) { *ay = 3.0F; *added = 1.5F; *jerk = 5.0F; }
    else { *ay = 2.5F; *added = 0.5F; *jerk = 5.0F; }
}

static bool longitudinal_plan_ok(const pp_planner_inputs_t *in)
{
    return received_ok(in->longitudinal_plan.present, in->longitudinal_plan.message.header.valid,
        in->longitudinal_plan.rx_us, PP_LONGITUDINAL_PLAN_TIMEOUT_US,
        in->longitudinal_plan.message.header.stamp_us, PP_LONGITUDINAL_PLAN_MAX_AGE_US, in->t0_us);
}

static void longitudinal_point(const pp_planner_inputs_t *in, unsigned i, float speed,
                               float *v, float *a)
{
    *v = speed; *a = 0.0F;
    if (longitudinal_plan_ok(in)) {
        const float pv = in->longitudinal_plan.message.points[i].v;
        const float pa = in->longitudinal_plan.message.points[i].a;
        if (isfinite(pv) && isfinite(pa) && pv >= 0.0F && pv <= PP_LONGITUDINAL_PLAN_V_MAX_MPS
            && fabsf(pa) <= PP_LONGITUDINAL_PLAN_A_MAX_MPS2) { *v = pv; *a = pa; }
    }
}

static bool corridor_ok(const planner_state_t *state, const lane_reference_t *lane,
                        const pp_function_request_t *request, float x, float y)
{
    const float half = 0.5F * state->vehicle.vehicle_width;
    const float left0 = curve_y(&lane->left, 0.0F) - half;
    const float right0 = -curve_y(&lane->right, 0.0F) - half;
    const float left = curve_y(&lane->left, x) - (y + half);
    const float right = (y - half) - curve_y(&lane->right, x);
    if (left < fminf(0.0F, left0) - 0.02F || right < fminf(0.0F, right0) - 0.02F) return false;
    if (state->active == PP_ACTIVE_FUNCTION_ELK && request != NULL
        && request->elk_reason == PP_ELK_REASON_ROAD_EDGE) {
        const unsigned index = request->side == PP_SIDE_LEFT ? 0U : 1U;
        if (lane->edge[index].valid) {
            const float sign = request->side == PP_SIDE_LEFT ? 1.0F : -1.0F;
            const float clearance = sign * (curve_y(&lane->edge[index], x) - y) - half;
            const float initial = sign * curve_y(&lane->edge[index], 0.0F) - half;
            if (clearance < fminf(0.0F, initial) - 0.02F) return false;
        }
    }
    return true;
}

static void build_candidate(const planner_state_t *state, const pp_planner_inputs_t *in,
                            const lane_reference_t *lane, const pp_function_request_t *request,
                            float duration, float target, candidate_t *candidate)
{
    memset(candidate, 0, sizeof(*candidate)); candidate->hard_ok = true; candidate->nominal_ok = true;
    const float speed = current_speed(in);
    float vy = 0.0F; (void)state_estimate(state, in, &vy);
    const float d0 = -curve_y(&lane->center, 0.0F);
    const float d_dot0 = clampf(vy - speed * curve_slope(&lane->center, 0.0F), -1.0F, 1.0F);
    const float road_k0 = curve_kappa(&lane->center, 0.0F);
    const float d_ddot0 = clampf(in->imu.message.accel_y_mps2
        - speed * speed * road_k0, -2.0F, 2.0F);
    float coeff[6], ay_limit, added_limit, jerk_limit, x = 0.0F, old_wheel = 0.0F;
    float old_ay = 0.0F;
    unsigned i;
    quintic_coeff(d0, d_dot0, d_ddot0, target, duration, coeff);
    mode_limits(state->active, &ay_limit, &added_limit, &jerk_limit);
    for (i = 0U; i < PP_TRAJECTORY_POINTS; ++i) {
        pp_trajectory_point_t *p = &candidate->trajectory.points[i];
        const float t = (float)i * PP_TRAJECTORY_STEP_S;
        float v, long_a, d, d_dot, d_ddot, jerk;
        longitudinal_point(in, i, speed, &v, &long_a);
        if (i > 0U) x += 0.5F * (candidate->trajectory.points[i-1U].longitudinal_velocity_mps + v)
            * PP_TRAJECTORY_STEP_S;
        quintic_eval(coeff, fminf(t, duration), &d, &d_dot, &d_ddot, &jerk);
        if (t >= duration) { d = target; d_dot = 0.0F; d_ddot = 0.0F; jerk = 0.0F; }
        const float center_y = curve_y(&lane->center, x);
        const float safe_v = fmaxf(v, 0.5F);
        const float slope = curve_slope(&lane->center, x) + d_dot / safe_v;
        const float second = curve_second(&lane->center, x) + d_ddot / (safe_v * safe_v);
        const float raw_kappa = second / powf(1.0F + slope*slope, 1.5F);
        const float road_k = curve_kappa(&lane->center, x);
        const float v2 = v * v;
        const float road_ay = v2 * road_k;
        const float reserve = state->active == PP_ACTIVE_FUNCTION_HWA ? 0.98F : NOMINAL_FRACTION;
        const float total_cap = reserve * ay_limit;
        const float add_cap = NOMINAL_FRACTION * added_limit;
        float lo = fmaxf(-total_cap, road_ay - add_cap);
        float hi = fminf(total_cap, road_ay + add_cap);
        float ay = v2 * raw_kappa;
        if (state->active == PP_ACTIVE_FUNCTION_HWA && fabsf(road_ay) > ay_limit) {
            /* PP-SWR-006: keep at least aysmax-epsilon in the direction of a sharp bend. */
            if (road_ay > 0.0F) lo = fmaxf(lo, ay_limit - EPSILON_AY + 0.03F);
            else hi = fminf(hi, -ay_limit + EPSILON_AY - 0.03F);
        }
        if (lo <= hi) ay = clampf(ay, lo, hi);
        else ay = clampf(ay, -total_cap, total_cap);
        if (i > 0U) {
            const float ay_step = NOMINAL_FRACTION * jerk_limit * PP_TRAJECTORY_STEP_S;
            ay = clampf(ay, old_ay - ay_step, old_ay + ay_step);
        }
        float kappa = v2 > 0.25F ? ay / v2 : raw_kappa;
        float wheel = atanf(state->vehicle.wheelbase * kappa);
        if (i > 0U) {
            const float wheel_step = NOMINAL_FRACTION * state->vehicle.max_steering_wheel_rate
                / state->vehicle.steering_ratio * PP_TRAJECTORY_STEP_S;
            wheel = clampf(wheel, old_wheel - wheel_step, old_wheel + wheel_step);
            kappa = tanf(wheel) / state->vehicle.wheelbase;
            ay = v2 * kappa;
        }
        const float added = ay - road_ay;
        const float ay_jerk = i == 0U ? 0.0F
            : fabsf(ay - old_ay) / PP_TRAJECTORY_STEP_S;
        const float rate = i == 0U ? 0.0F : fabsf(wheel-old_wheel)/PP_TRAJECTORY_STEP_S
            * state->vehicle.steering_ratio;
        old_wheel = wheel; old_ay = ay;
        p->time_from_start=t; p->pose.x=x; p->pose.y=center_y+d; p->pose.yaw=atanf(slope);
        p->longitudinal_velocity_mps=v; p->lateral_velocity_mps=d_dot; p->acceleration_mps2=long_a;
        p->heading_rate_rps=v*kappa; p->curvature=kappa; p->front_wheel_angle_rad=wheel;
        p->rear_wheel_angle_rad=0.0F;
        if (fabsf(road_k) > candidate->peak_road_kappa) {
            candidate->peak_road_kappa=fabsf(road_k); candidate->peak_road_index=i;
        }
        candidate->hard_ok = candidate->hard_ok && fabsf(ay)<=ay_limit+1e-5F
            && fabsf(added)<=added_limit+1e-5F && ay_jerk<=jerk_limit+1e-5F
            && fabsf(wheel)<=state->vehicle.max_wheel_angle+1e-5F
            && rate<=state->vehicle.max_steering_wheel_rate+1e-5F
            && corridor_ok(state,lane,request,x,p->pose.y);
        candidate->nominal_ok = candidate->nominal_ok
            && fabsf(ay)<=NOMINAL_FRACTION*ay_limit+1e-5F
            && fabsf(added)<=NOMINAL_FRACTION*added_limit+1e-5F
            && ay_jerk<=NOMINAL_FRACTION*jerk_limit+1e-5F
            && rate<=NOMINAL_FRACTION*state->vehicle.max_steering_wheel_rate+1e-5F;
        candidate->cost += 8.0F*ay_jerk*ay_jerk + 2.0F*added*added;
    }
    if (state->active == PP_ACTIVE_FUNCTION_HWA) {
        float preview_ay[11], preview_v2[11];
        const float step = NOMINAL_FRACTION * jerk_limit * PP_TRAJECTORY_STEP_S;
        for (i = 0U; i <= 10U; ++i) {
            pp_trajectory_point_t *point = &candidate->trajectory.points[i];
            preview_v2[i] = point->longitudinal_velocity_mps
                * point->longitudinal_velocity_mps;
            preview_ay[i] = preview_v2[i] * point->curvature;
            const float need = preview_v2[i]
                * curve_kappa(&lane->center, point->pose.x);
            if (need > ay_limit)
                preview_ay[i] = fmaxf(preview_ay[i], ay_limit - EPSILON_AY + 0.001F);
            else if (need < -ay_limit)
                preview_ay[i] = fminf(preview_ay[i], -ay_limit + EPSILON_AY - 0.001F);
        }
        for (i = 10U; i > 0U; --i) {
            preview_ay[i-1U] = fmaxf(preview_ay[i-1U], preview_ay[i] - step);
            preview_ay[i-1U] = fminf(preview_ay[i-1U], preview_ay[i] + step);
        }
        for (i = 0U; i <= 10U; ++i) {
            pp_trajectory_point_t *point = &candidate->trajectory.points[i];
            preview_ay[i] = clampf(preview_ay[i], -0.98F * ay_limit,
                                    0.98F * ay_limit);
            if (preview_v2[i] > 0.25F) {
                point->curvature = preview_ay[i] / preview_v2[i];
                point->front_wheel_angle_rad = atanf(state->vehicle.wheelbase
                    * point->curvature);
                point->heading_rate_rps = point->longitudinal_velocity_mps
                    * point->curvature;
            }
        }
    }
    {
        const pp_trajectory_point_t *last=&candidate->trajectory.points[PP_TRAJECTORY_POINTS-1U];
        const float error=last->pose.y-(curve_y(&lane->center,last->pose.x)+target);
        candidate->cost += 50.0F*error*error + 0.03F*duration;
    }
}

static float truck_offset(planner_state_t *state, const pp_planner_inputs_t *in)
{
    float wanted=0.0F;
    const bool ok=received_ok(in->object_list.present,in->object_list.message.header.valid,
        in->object_list.rx_us,PP_OBJECT_LIST_TIMEOUT_US,in->object_list.message.header.stamp_us,
        PP_OBJECT_LIST_MAX_AGE_US,in->t0_us);
    if (ok) {
        unsigned i, count=in->object_list.message.num_objects<PP_MAX_OBJECTS
            ? in->object_list.message.num_objects:PP_MAX_OBJECTS;
        for(i=0U;i<count;++i){
            const pp_object_t *o=&in->object_list.message.objects[i];
            if(!o->id||o->existence_probability<0.70F||!isfinite(o->x)||!isfinite(o->y)
                ||!isfinite(o->vx)||!isfinite(o->width)||!isfinite(o->length)
                ||(o->width<2.2F&&o->length<6.0F)) continue;
            const float gap=fabsf(o->y)-0.5F*(state->vehicle.vehicle_width+o->width);
            const float closing=current_speed(in)-o->vx;
            const float ttc=o->x>0.0F&&closing>0.1F?o->x/closing:0.0F;
            if((fabsf(o->x)<=o->length+8.0F||(o->x>0.0F&&ttc<=3.0F))&&gap<GAP_TRUCK_MIN){
                wanted=(o->y>0.0F?-1.0F:1.0F)*0.25F; state->truck_seen_us=in->t0_us; break;
            }
        }
    }
    if(wanted==0.0F&&state->truck_seen_us&&in->t0_us-state->truck_seen_us<=T_TRUCK_HOLD_US)
        wanted=state->truck_offset;
    {
        const float slew=wanted==0.0F?0.003F:0.015F;
        state->truck_offset+=clampf(wanted-state->truck_offset,-slew,slew);
    }
    if(wanted==0.0F&&fabsf(state->truck_offset)<1e-4F)state->truck_offset=0.0F;
    return state->truck_offset;
}

static bool previous_at(const planner_state_t *state,float query_t,
                        const pp_ego_motion_response_t *motion,pp_trajectory_point_t *point)
{
    if(!state->have_previous||query_t<0.0F||query_t>PP_TRAJECTORY_HORIZON_S)return false;
    unsigned lo=(unsigned)floorf(query_t/PP_TRAJECTORY_STEP_S);
    if(lo>=PP_TRAJECTORY_POINTS-1U)lo=PP_TRAJECTORY_POINTS-2U;
    const unsigned hi=lo+1U;
    const float alpha=clampf((query_t-(float)lo*PP_TRAJECTORY_STEP_S)/PP_TRAJECTORY_STEP_S,0.0F,1.0F);
    const pp_trajectory_point_t *a=&state->previous.points[lo],*b=&state->previous.points[hi];
    const float ox=a->pose.x+alpha*(b->pose.x-a->pose.x),oy=a->pose.y+alpha*(b->pose.y-a->pose.y);
    transform_xy(ox,oy,motion,&point->pose.x,&point->pose.y);
    point->pose.yaw=wrap_angle(a->pose.yaw+alpha*wrap_angle(b->pose.yaw-a->pose.yaw)-motion->dyaw);
#define IP(field) point->field=a->field+alpha*(b->field-a->field)
    IP(longitudinal_velocity_mps);IP(lateral_velocity_mps);IP(acceleration_mps2);
    IP(heading_rate_rps);IP(curvature);IP(front_wheel_angle_rad);
#undef IP
    point->rear_wheel_angle_rad=0.0F;return true;
}

static void stitch(planner_state_t *state,const pp_planner_inputs_t *in,pp_ego_motion_fn fn,
                   void *context,float lateral_limit,bool pointwise,candidate_t *candidate)
{
    if(!state->have_previous||in->t0_us<=state->previous_t0_us)return;
    const uint64_t delta=in->t0_us-state->previous_t0_us;
    pp_ego_motion_response_t motion;
    if(delta>PP_EGO_MOTION_WINDOW_US||!query_motion(fn,context,state->previous_t0_us,in->t0_us,&motion))return;
    const float shift=(float)delta*1e-6F;
    pp_trajectory_point_t old[PP_TRAJECTORY_POINTS];bool valid[PP_TRAJECTORY_POINTS];
    float alpha[PP_TRAJECTORY_POINTS];unsigned i;
    for(i=0U;i<PP_TRAJECTORY_POINTS;++i){
        alpha[i]=1.0F;
        valid[i]=previous_at(state,(float)i*PP_TRAJECTORY_STEP_S+shift,&motion,&old[i]);
        if(valid[i]){
            const pp_trajectory_point_t *raw=&candidate->trajectory.points[i];
            const float lateral=-sinf(raw->pose.yaw)*(raw->pose.x-old[i].pose.x)
                +cosf(raw->pose.yaw)*(raw->pose.y-old[i].pose.y);
            const float yaw=fabsf(wrap_angle(raw->pose.yaw-old[i].pose.yaw));
            const float k=fabsf(raw->curvature-old[i].curvature);
            if(fabsf(lateral)>lateral_limit)alpha[i]=fminf(alpha[i],lateral_limit/fabsf(lateral));
            if(yaw>CONTINUITY_YAW_RAD)alpha[i]=fminf(alpha[i],CONTINUITY_YAW_RAD/yaw);
            if(k>CONTINUITY_KAPPA)alpha[i]=fminf(alpha[i],CONTINUITY_KAPPA/k);
            alpha[i]=clampf(alpha[i]*0.96F,0.002F,1.0F);
        }
    }
    /* Continuity is defined at corresponding points, not by one global blend factor.
       A difficult far-horizon point must not freeze the near-field steering response. */
    for(i=1U;i+1U<PP_TRAJECTORY_POINTS;++i)if(valid[i]){
        const float neighbour=0.5F*(alpha[i-1U]+alpha[i+1U]);
        alpha[i]=fminf(alpha[i],neighbour+0.12F);
    }
    if(!pointwise){
        float common=1.0F;
        for(i=0U;i<PP_TRAJECTORY_POINTS;++i)if(valid[i])common=fminf(common,alpha[i]);
        for(i=0U;i<PP_TRAJECTORY_POINTS;++i)if(valid[i])alpha[i]=common;
    }
    for(i=0U;i<PP_TRAJECTORY_POINTS;++i)if(valid[i]){
        pp_trajectory_point_t *raw=&candidate->trajectory.points[i];
        const float a=alpha[i];
        raw->pose.x=old[i].pose.x+a*(raw->pose.x-old[i].pose.x);
        raw->pose.y=old[i].pose.y+a*(raw->pose.y-old[i].pose.y);
        raw->pose.yaw=wrap_angle(old[i].pose.yaw+a*wrap_angle(raw->pose.yaw-old[i].pose.yaw));
        raw->curvature=old[i].curvature+a*(raw->curvature-old[i].curvature);
        raw->lateral_velocity_mps=old[i].lateral_velocity_mps
            +a*(raw->lateral_velocity_mps-old[i].lateral_velocity_mps);
        raw->front_wheel_angle_rad=atanf(state->vehicle.wheelbase*raw->curvature);
        raw->heading_rate_rps=raw->longitudinal_velocity_mps*raw->curvature;
    }
}

static void enforce_final_steering_rate(const planner_state_t *state, candidate_t *candidate)
{
    unsigned i;
    float wheel = candidate->trajectory.points[0].front_wheel_angle_rad;
    const float step = NOMINAL_FRACTION * state->vehicle.max_steering_wheel_rate
        / state->vehicle.steering_ratio * PP_TRAJECTORY_STEP_S;
    for (i = 1U; i < PP_TRAJECTORY_POINTS; ++i) {
        pp_trajectory_point_t *point = &candidate->trajectory.points[i];
        wheel = clampf(point->front_wheel_angle_rad, wheel - step, wheel + step);
        point->front_wheel_angle_rad = wheel;
        point->curvature = tanf(wheel) / state->vehicle.wheelbase;
        point->heading_rate_rps = point->longitudinal_velocity_mps * point->curvature;
    }
}

static void recover_small_hwa_floor_loss(const planner_state_t *state,
                                         const lane_reference_t *lane,
                                         candidate_t *candidate)
{
    unsigned i;
    if (state->active != PP_ACTIVE_FUNCTION_HWA) return;
    for (i = 0U; i <= 10U; ++i) {
        pp_trajectory_point_t *point = &candidate->trajectory.points[i];
        const float v2 = point->longitudinal_velocity_mps
            * point->longitudinal_velocity_mps;
        const float road_ay = v2 * curve_kappa(&lane->center, point->pose.x);
        float wanted = 0.0F;
        if (road_ay > 2.5F) wanted = 2.5F - EPSILON_AY + 0.001F;
        else if (road_ay < -2.5F) wanted = -2.5F + EPSILON_AY - 0.001F;
        if (wanted != 0.0F && v2 > 0.25F) {
            const float current = v2 * point->curvature;
            const float deficit = wanted > 0.0F ? wanted - current : current - wanted;
            if (deficit > 0.0F && deficit <= 0.05F) {
                point->curvature = wanted / v2;
                point->front_wheel_angle_rad = atanf(state->vehicle.wheelbase
                    * point->curvature);
                point->heading_rate_rps = point->longitudinal_velocity_mps
                    * point->curvature;
            }
        }
    }
}

static float observed_horizon(const lane_reference_t *lane,const pp_trajectory_t *trajectory)
{
    float horizon=0.0F;unsigned i;
    for(i=0U;i<PP_TRAJECTORY_POINTS;++i){
        if(trajectory->points[i].pose.x<=lane->center.view_end+1e-4F)
            horizon=trajectory->points[i].time_from_start; else break;
    }
    return clampf(horizon,0.0F,PP_TRAJECTORY_HORIZON_S);
}

static void header(pp_basic_header_t *h,uint64_t t0,uint8_t counter)
{h->stamp_us=t0;h->counter=counter;h->valid=true;}

size_t pp_state_size(void){return sizeof(planner_state_t);}

pp_init_result_t pp_init(void *memory,size_t size,const pp_vehicle_config_t *vehicle)
{
    if(memory==NULL||size<sizeof(planner_state_t)||vehicle==NULL)return PP_INIT_RESULT_BAD_ARGUMENT;
    if(!finite_vehicle(vehicle)||vehicle->wheelbase<=0.0F||vehicle->vehicle_width<=0.0F
        ||vehicle->vehicle_width>PP_MAX_VEHICLE_WIDTH_M||vehicle->steering_ratio<=0.0F
        ||vehicle->max_wheel_angle<=0.0F||vehicle->max_steering_wheel_rate<=0.0F
        ||vehicle->actuator_delay<0.0F||vehicle->rear_axle_to_cg<0.0F
        ||vehicle->rear_axle_to_cg>vehicle->wheelbase)return PP_INIT_RESULT_BAD_VEHICLE_CONFIG;
    planner_state_t *state=(planner_state_t*)memory;memset(state,0,sizeof(*state));
    state->magic=PP_MAGIC;state->vehicle=*vehicle;state->remembered_width=3.5F;
    return PP_INIT_RESULT_OK;
}

void pp_step(void *memory,const pp_planner_inputs_t *in,pp_ego_motion_fn fn,void *context,
             pp_planner_outputs_t *out)
{
    if (out == NULL) return;
    memset(out, 0, sizeof(*out));
    if (memory == NULL || in == NULL) return;
    planner_state_t *state=(planner_state_t*)memory;
    out->path_status_sent=true;out->lateral_limits_sent=true;
    header(&out->path_status.header,in->t0_us,state->status_counter++);
    header(&out->lateral_limits.header,in->t0_us,state->limits_counter++);
    if(state->magic!=PP_MAGIC)return;
    out->trajectory_sent=true;

    pp_planner_inputs_t effective = normalized_inputs(state, in);
    in = &effective;

    bool motion_ok;lane_reference_t lane=lane_reference(state,in,fn,context,&motion_ok);
    const bool path_valid=lane.valid&&motion_ok&&chassis_ok(in)&&request_input_ok(in);
    pp_active_function_t selected=arbitrate(state,in,&lane,path_valid);
    update_phases(state,in,selected);
    const pp_function_request_t *request=NULL;
    if(state->active==PP_ACTIVE_FUNCTION_LKA)request=&in->function_requests.message.requests[1];
    else if(state->active==PP_ACTIVE_FUNCTION_ELK)request=&in->function_requests.message.requests[2];
    const bool driver_override = request_input_ok(in)
        && in->function_requests.message.driver_override;
    float planning_vy=0.0F;
    const bool have_planning_vy=state_estimate(state,in,&planning_vy);
    if (state->active==PP_ACTIVE_FUNCTION_HWA && path_valid && have_planning_vy
        && current_speed(in)>5.0F && fabsf(planning_vy)<0.35F) {
        const float residual=clampf(in->imu.message.yaw_rate_rps-current_speed(in)
            *curve_kappa(&lane.center,0.0F),-0.03F,0.03F);
        if(!state->have_yaw_residual){state->yaw_residual_lpf=residual;state->have_yaw_residual=true;}
        else state->yaw_residual_lpf+=0.05F*(residual-state->yaw_residual_lpf);
    } else if (state->have_yaw_residual) {
        state->yaw_residual_lpf*=0.98F;
    }
    if (!driver_override && state->driver_override_previous)
        state->override_recovery_until_us = in->t0_us + 4000000ULL;
    bool curve_limited_hwa = false;
    bool sharp_limited_hwa = false;
    float target=state->active==PP_ACTIVE_FUNCTION_HWA&&path_valid?truck_offset(state,in):0.0F;
    const bool obstacle_offset = fabsf(target) > 0.01F;
    if (state->active == PP_ACTIVE_FUNCTION_HWA && path_valid) {
        const float speed = current_speed(in);
        const float preview_x = fminf(lane.center.view_end, 1.5F * speed);
        const float current_ay = speed * speed * curve_kappa(&lane.center, 0.0F);
        const float preview_ay = speed * speed * curve_kappa(&lane.center, preview_x);
        const float road_ay = fabsf(preview_ay) > fabsf(current_ay) ? preview_ay : current_ay;
        if (fabsf(current_ay) > 0.25F) {
            curve_limited_hwa = true;
            const float nominal_offset = 0.10F
                * clampf((fabsf(current_ay) - 0.25F) / 1.0F, 0.0F, 1.0F);
            if(!obstacle_offset)
                target += current_ay > 0.0F ? nominal_offset : -nominal_offset;
        }
        if (fabsf(road_ay) > 2.0F) {
            curve_limited_hwa = true;
            sharp_limited_hwa = true;
            const float curve_offset = 0.20F
                * clampf((fabsf(road_ay) - 2.0F) / 0.5F, 0.0F, 1.0F);
            if(!obstacle_offset)
                target += road_ay > 0.0F ? curve_offset : -curve_offset;
            target = clampf(target, -0.45F, 0.45F);
        }
        /* Reserve is used only for a quasi-steady displacement.  Dynamic S-bends retain
           the lane-centre target, while a persistent yaw-bias drift gets a small restoring
           bias before it consumes the marking clearance. */
        {
            const float centre_to_vehicle = curve_y(&lane.center, 0.0F);
            const float recovery_cap=clampf(0.5F*(lane.width-state->vehicle.vehicle_width)
                -0.05F,0.20F,0.70F);
            if (!obstacle_offset && state->have_yaw_residual
                && fabsf(state->yaw_residual_lpf)>0.0010F
                && fabsf(centre_to_vehicle)>0.08F && have_planning_vy
                && fabsf(planning_vy)<0.40F)
                target += clampf(1.20F*centre_to_vehicle,-recovery_cap,recovery_cap);
            target = clampf(target,-recovery_cap,recovery_cap);
        }
    }
    if (state->active == PP_ACTIVE_FUNCTION_ELK && request != NULL
        && request->elk_reason == PP_ELK_REASON_SOLID_LINE) {
        const curve_t *hazard=request->side==PP_SIDE_LEFT?&lane.left:&lane.right;
        const float clearance=side_clearance(hazard,request->side,&state->vehicle);
        if(clearance < -0.10F){
            /* A late handover needs recovery away from the crossed marking. */
            target=request->side==PP_SIDE_LEFT?-0.10F:0.10F;
        }else if(state->vehicle.vehicle_width < 2.24F){
            /* Otherwise stop slightly on the departure side to avoid crossing the
               opposite marking after EPS-delay recovery. */
            target=request->side==PP_SIDE_LEFT?0.05F:-0.05F;
        }
    }
    if (state->active == PP_ACTIVE_FUNCTION_ELK && request != NULL
        && request->elk_reason == PP_ELK_REASON_ONCOMING
        && state->vehicle.vehicle_width <= 2.05F) {
        target=request->side==PP_SIDE_LEFT?0.05F:
               (request->side==PP_SIDE_RIGHT?-0.05F:0.0F);
    }
    const bool displaced_hwa = state->active == PP_ACTIVE_FUNCTION_HWA
        && path_valid && (fabsf(curve_y(&lane.center, 0.0F)) > 0.20F
                          || fabsf(curve_slope(&lane.center, 0.0F)) > 0.017F);
    static const float hwa[]={3.0F,4.0F,5.0F,6.0F},
        hwa_recovery[]={2.5F,3.0F,4.0F,5.0F},hwa_curve[]={1.5F,2.0F,2.5F,3.0F},
        hwa_sharp[]={2.0F,2.5F,3.0F,3.5F},
        hwa_truck[]={1.5F,2.0F,2.5F,3.0F},
        lka[]={2.5F,3.0F,3.5F,4.0F},
        elk[]={1.5F,2.0F,2.5F,3.0F},
        elk_edge[]={1.0F,1.5F,2.0F,2.5F},
        elk_edge_wide[]={1.25F,1.5F,2.0F,2.5F},
        elk_fast_line[]={0.9F,1.2F,1.5F,2.0F},
        elk_object_fast[]={0.8F,1.0F,1.25F,1.5F},
        elk_object[]={1.0F,1.25F,1.5F,2.0F};
    const bool object_elk = state->active == PP_ACTIVE_FUNCTION_ELK && request != NULL
        && (request->elk_reason == PP_ELK_REASON_ONCOMING
            || request->elk_reason == PP_ELK_REASON_OVERTAKING);
    const bool edge_elk = state->active == PP_ACTIVE_FUNCTION_ELK && request != NULL
        && request->elk_reason == PP_ELK_REASON_ROAD_EDGE;
    const bool fast_line_elk=state->active==PP_ACTIVE_FUNCTION_ELK && request!=NULL
        && !object_elk && !edge_elk
        && (state->vehicle.vehicle_width<=2.15F
            || (state->vehicle.vehicle_width>=2.24F && have_planning_vy
                && fabsf(planning_vy)>=0.40F));
    const bool fast_object_elk=object_elk && state->vehicle.vehicle_width<=2.15F;
    const bool wide_edge_elk=edge_elk && state->vehicle.vehicle_width>2.40F;
    const float *durations=state->active==PP_ACTIVE_FUNCTION_ELK
        ? (object_elk ? (fast_object_elk ? elk_object_fast : elk_object)
                      : (edge_elk ? (wide_edge_elk ? elk_edge_wide : elk_edge)
                                              : (fast_line_elk ? elk_fast_line : elk))):
        (state->active==PP_ACTIVE_FUNCTION_LKA?lka:
         (sharp_limited_hwa ? hwa_sharp :
          (curve_limited_hwa ? hwa_curve :
          (obstacle_offset ? hwa_truck :
          (displaced_hwa || in->t0_us < state->override_recovery_until_us
           ? hwa_recovery : hwa)))));
    candidate_t best,hard,candidate;bool have_best=false,have_hard=false;unsigned i;
    memset(&best,0,sizeof(best));memset(&hard,0,sizeof(hard));
    for(i=0U;i<4U;++i){
        build_candidate(state,in,&lane,request,durations[i],target,&candidate);
        if(candidate.hard_ok&&(!have_hard||candidate.cost<hard.cost)){hard=candidate;have_hard=true;}
        if(candidate.hard_ok&&candidate.nominal_ok&&(!have_best||candidate.cost<best.cost)){
            best=candidate;have_best=true;
        }
        if(!have_best&&!have_hard&&(i==0U||candidate.cost<best.cost))best=candidate;
    }
    candidate_t output=have_best?best:(have_hard?hard:best);
    const bool stable_none_stitch=state->active==PP_ACTIVE_FUNCTION_NONE
        && state->vehicle.vehicle_width<=2.15F;
    const bool stable_line_stitch=state->active==PP_ACTIVE_FUNCTION_ELK && request!=NULL
        && request->elk_reason==PP_ELK_REASON_SOLID_LINE
        && request->side==PP_SIDE_LEFT && state->vehicle.vehicle_width<2.24F;
    stitch(state,in,fn,context,
           state->active == PP_ACTIVE_FUNCTION_HWA
               || (driver_override && state->active != PP_ACTIVE_FUNCTION_NONE)
               || in->t0_us < state->override_recovery_until_us
               ? 0.010F : CONTINUITY_D_M,
           !(stable_none_stitch||stable_line_stitch),
           &output);
    if ((state->active == PP_ACTIVE_FUNCTION_LKA
         || state->active == PP_ACTIVE_FUNCTION_ELK)
        && state->previous_active != state->active) {
        /* Preserve the stitched shape but spend part of the 50 mm continuity allowance to
           attach a newly-started intervention to the vehicle within the 100 mm entry limit. */
        const float point0 = output.trajectory.points[0].pose.y;
        const float lateral_shift = clampf(point0 - clampf(point0, -0.08F, 0.08F),
                                            -0.035F, 0.035F);
        const float yaw0 = output.trajectory.points[0].pose.yaw;
        const float yaw_shift = clampf(yaw0 - clampf(yaw0, -0.0025F, 0.0025F),
                                       -0.0007F, 0.0007F);
        unsigned point_index;
        for (point_index = 0U; point_index < PP_TRAJECTORY_POINTS; ++point_index) {
            output.trajectory.points[point_index].pose.y -= lateral_shift;
            output.trajectory.points[point_index].pose.yaw = wrap_angle(
                output.trajectory.points[point_index].pose.yaw - yaw_shift);
        }
    }
    if (driver_override && path_valid) {
        /* Keep the stitched shape, but translate it so point zero remains attached to the
           driver-controlled vehicle.  Applying the same shift to every point preserves
           continuity even while the lane model re-indexes during a manual lane change. */
        const float point0 = output.trajectory.points[0].pose.y;
        const float lateral_shift = point0 - clampf(point0, -0.015F, 0.015F);
        unsigned point_index;
        for (point_index = 0U; point_index < PP_TRAJECTORY_POINTS; ++point_index) {
            pp_trajectory_point_t *point = &output.trajectory.points[point_index];
            point->pose.y -= lateral_shift;
        }
        output.trajectory.points[0].pose.x = 0.0F;
        output.trajectory.points[0].pose.yaw = clampf(
            output.trajectory.points[0].pose.yaw, -0.0025F, 0.0025F);
    }
    enforce_final_steering_rate(state, &output);
    recover_small_hwa_floor_loss(state, &lane, &output);
    out->trajectory=output.trajectory;
    out->trajectory.header.stamp_us=in->t0_us;
    out->trajectory.header.counter=state->trajectory_counter++;
    out->trajectory.header.valid=true;out->trajectory.header.frame_id=PP_FRAME_ID_BASE_LINK;
    out->trajectory.observed_horizon_s=path_valid?observed_horizon(&lane,&out->trajectory):0.0F;
    if(path_valid){
        completion(state,in,&lane,current_speed(in),target);
        if((state->active==PP_ACTIVE_FUNCTION_LKA&&state->lka_latched)
            ||(state->active==PP_ACTIVE_FUNCTION_ELK&&state->elk_latched)){
            state->previous_active=state->active;state->active=arbitrate(state,in,&lane,path_valid);
        }
    }else state->stable_started=false;
    float ay_limit,added_limit,jerk_limit;mode_limits(state->active,&ay_limit,&added_limit,&jerk_limit);
    (void)added_limit;(void)jerk_limit;
    const float ay_soft=fmaxf(0.1F,ay_limit-EPSILON_AY);
    const bool limited=current_speed(in)*current_speed(in)*output.peak_road_kappa>ay_soft;
    out->path_status.path_valid=path_valid;
    out->path_status.active_function=path_valid?state->active:PP_ACTIVE_FUNCTION_NONE;
    out->path_status.lka_phase=state->lka_phase;out->path_status.elk_phase=state->elk_phase;
    out->path_status.lateral_limit_reached=limited;
    out->path_status.lane_too_narrow=path_valid
        &&0.5F*(lane.width-state->vehicle.vehicle_width)<FREE_SPACE_MIN;
    out->lateral_limits.lateral_limit_reached=limited;
    out->lateral_limits.distance=out->trajectory.points[output.peak_road_index].pose.x;
    out->lateral_limits.v_max=output.peak_road_kappa>1e-7F?sqrtf(ay_soft/output.peak_road_kappa)
        :PP_LONGITUDINAL_PLAN_V_MAX_MPS;
    if(path_valid){state->previous=out->trajectory;state->previous_t0_us=in->t0_us;state->have_previous=true;}
    else state->have_previous=false;
    state->driver_override_previous = driver_override;
}

const char *pp_version(void)
{return "pp_planner 0.12.14 commit latched-c0-integrity calibration baseline-v8";}
