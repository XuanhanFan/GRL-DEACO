from __future__ import annotations
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import numpy as np
logger = logging.getLogger(__name__)
DEFAULT_NORMALIZATION_RANGES = {'E_op': (0, 50000), 'CO2_op': (0, 25000), 'CO2_emb': (0, 1000), 'L': (0, 1000), 'N_bend': (0, 100), 'alt': (0, 50), 'viol': (0, 100), 'f_Energy': (0, 5000), 'f_Install': (0, 2000), 'f_Height': (0, 500)}

@dataclass
class DEACOParameters:
    M_ants: int = 30
    K_iterations: int = 60
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.1
    Q: float = 100.0
    tau_0: float = 0.1
    tau_max0: float = 10.0
    tau_min0: float = 0.01
    A_q0: float = 0.8
    B_q0: float = 0.1
    delta_gamma: float = 0.9
    omega_Length: float = 1.0
    omega_Bend: float = 4.0
    omega_Energy: float = 0.5
    omega_Install: float = 1.5
    omega_direction_reward: float = 2.5
    omega_bend_penalty: float = 2.0
    omega_height_penalty: float = 1.0
    kappa_y: float = 1.0
    delta_xz: float = 10.0
    s_sigmoid: float = 3.0
    use_anisotropic_height_weight: bool = True
    max_steps: int = 20000
    early_stop_patience: int = 20
    early_stop_threshold: float = 0.05
    fluid_density: float = 998.0
    gravity: float = 9.80665
    flow_rate: float = 0.0278
    pipe_diameter: float = 0.154
    pipe_roughness: float = 4.5e-05
    dynamic_viscosity: float = 0.001002
    darcy_friction: float = None
    bend_loss_K: float = 0.9
    pump_efficiency: float = 0.75
    annual_hours: float = 8000.0
    grid_carbon_intensity: float = 0.51
    reference_energy: float = 10000.0
    epsilon: float = 1e-06
    pipe_carbon_factor: float = 5.0
    elbow_carbon_factor: float = 2.0
    reference_pipe_diameter: float = 0.154
    pipe_carbon_scale_exponent: float = 2.0
    w_op: float = 3.0
    w_emb: float = 2.0
    w_L: float = 1.0
    w_bend: float = 1.5
    w_alt: float = 1.0
    w_clear: float = 2.0
    normalization_mode: str = 'adaptive'
    normalization_guard_band: float = 0.15
    normalization_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scene_normalization_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    lambda_1: float = 1.0
    lambda_2: float = 2.0
    lambda_3: float = 1.5
    lambda_loc: float = 1.0
    y_ref: float = 0.0
    use_hybrid_normalization: bool = False
    omega_Energy_raw: float = 0.0001
    omega_Install_raw: float = 0.00075
    omega_height_penalty_raw: float = 0.002

def create_preset_params(medium_type='water', pipe_condition='new_steel', operation_mode='continuous', use_hybrid_normalization=False, verbose=True):
    params = DEACOParameters()
    if medium_type == 'water':
        params.fluid_density = 998.0
        params.dynamic_viscosity = 0.001002
        params.flow_rate = 0.0278
    elif medium_type == 'crude_oil':
        params.fluid_density = 875.0
        params.dynamic_viscosity = 0.01
        params.flow_rate = 0.42
    elif medium_type == 'refined_oil':
        params.fluid_density = 550.0
        params.dynamic_viscosity = 0.001
        params.flow_rate = 0.22
    elif medium_type == 'chemical':
        params.fluid_density = 850.0
        params.dynamic_viscosity = 0.002
        params.flow_rate = 0.14
    elif medium_type == 'circulating_water':
        params.fluid_density = 998.0
        params.dynamic_viscosity = 0.001002
        params.flow_rate = 0.69
    else:
        logger.debug("Unknown medium type %s; using default water-like parameters.", medium_type)
    if pipe_condition == 'new_steel':
        params.pipe_roughness = 4.5e-05
        params.darcy_friction = None
    elif pipe_condition == 'old_steel':
        params.pipe_roughness = 0.00015
        params.darcy_friction = None
    elif pipe_condition == 'plastic':
        params.pipe_roughness = 1e-05
        params.darcy_friction = None
    else:
        logger.debug("Unknown pipe condition %s; using default pipe roughness.", pipe_condition)
    if operation_mode == 'continuous':
        params.annual_hours = 8000.0
    elif operation_mode == 'intermittent':
        params.annual_hours = 6000.0
    else:
        logger.debug("Unknown operation mode %s; using default annual hours.", operation_mode)
    if medium_type in ['crude_oil', 'refined_oil']:
        params.pipe_carbon_factor = 6.0
        params.elbow_carbon_factor = 2.5
    else:
        params.pipe_carbon_factor = 5.0
        params.elbow_carbon_factor = 2.0
    if medium_type in ['crude_oil']:
        params.w_op = 3.5
        params.w_emb = 2.0
    elif medium_type == 'circulating_water':
        params.w_op = 2.5
        params.w_L = 1.5
        params.w_bend = 2.0
    params.use_hybrid_normalization = use_hybrid_normalization
    if verbose:
        logger.debug(
            "Created DEACO preset: medium=%s, pipe_condition=%s, operation=%s, flow_rate=%.6g, pipe_diameter=%.6g",
            medium_type,
            pipe_condition,
            operation_mode,
            params.flow_rate,
            params.pipe_diameter,
        )
    return params

def validate_parameters(params, verbose=True):
    warnings = []
    if not 450 <= params.fluid_density <= 1000:
        warnings.append(f'WARNING fluid density {params.fluid_density} kg/m3 is outside the typical range (450-1000 kg/m3)')
    flow_rate_m3h = params.flow_rate * 3600
    if not 50 <= flow_rate_m3h <= 5000:
        warnings.append(f'WARNING flow rate {flow_rate_m3h:.0f} m3/h is outside the typical range (50-5000 m3/h)')
    if not 0.053 <= params.pipe_diameter <= 0.788:
        warnings.append(f'WARNING pipe diameter {params.pipe_diameter} m is outside the typical range (DN50-DN800)')
    if not 0.65 <= params.pump_efficiency <= 0.85:
        warnings.append(f'WARNING pump efficiency {params.pump_efficiency} is outside the typical range (0.65-0.85)')
    if params.annual_hours not in [6000.0, 8000.0]:
        if not 5000 <= params.annual_hours <= 8760:
            warnings.append(f'WARNING annual operating hours {params.annual_hours} h should usually be 6000 or 8000')
    if not 0.4 <= params.grid_carbon_intensity <= 0.8:
        warnings.append(f'WARNING grid carbon intensity {params.grid_carbon_intensity} kgCO2/kWh is outside the typical range (0.4-0.8)')
    total_weight = params.w_op + params.w_emb + params.w_L + params.w_bend + params.w_alt + params.w_clear
    if total_weight < 5.0 or total_weight > 15.0:
        warnings.append(f'WARNING objective weight sum {total_weight:.1f} is outside the suggested range (5.0-15.0)')
    is_valid = len(warnings) == 0
    if verbose:
        if warnings:
            for w in warnings:
                logger.warning(w)
        else:
            logger.debug("DEACO parameter validation passed.")
    return (is_valid, warnings)

def initialize_scene_normalization_ranges(params, grid_info, connections):
    try:
        pitch = grid_info.get('pitch', 0.1)
        bounds = grid_info.get('bounds', {})
        area_x = bounds.get('x', (0.0, 0.0))
        area_y = bounds.get('y', (0.0, 0.0))
        area_z = bounds.get('z', (0.0, 0.0))
        area_x = area_x[1] - area_x[0]
        area_y = area_y[1] - area_y[0]
        area_z = area_z[1] - area_z[0]
        scene_diag = np.linalg.norm([area_x, area_y, area_z])
        max_manhattan = 0.0
        max_vertical = 0.0
        for conn in connections:
            start = np.array(conn['from_pos'])
            end = np.array(conn['to_pos'])
            diff = np.abs(end - start)
            max_manhattan = max(max_manhattan, diff.sum())
            max_vertical = max(max_vertical, diff[1])
        L_max_steps = max(max_manhattan / max(pitch, 1e-06), 20.0) * 1.2
        N_bend_max = max(min(L_max_steps, 200.0), 10.0)
        alt_max = max(max_vertical, area_y) * 1.2
        f_energy_max = L_max_steps * max(area_x, area_y, area_z, 1.0) * 0.5
        f_install_max = max(scene_diag, 1.0)
        viol_max = max(area_x, area_z, 1.0) * max(L_max_steps * 0.1, 1.0)
        f_height_max = alt_max + area_y * 0.5
        scene_ranges = {'L': (0.0, L_max_steps), 'N_bend': (0.0, N_bend_max), 'alt': (0.0, alt_max), 'viol': (0.0, viol_max), 'f_Energy': (0.0, f_energy_max), 'f_Install': (0.0, f_install_max), 'f_Height': (0.0, f_height_max)}
        if not hasattr(params, 'scene_normalization_ranges') or params.scene_normalization_ranges is None:
            params.scene_normalization_ranges = {}
        for key, value in scene_ranges.items():
            params.scene_normalization_ranges[key] = list(value)
    except Exception as e:
        logger.debug("Failed to initialize scene normalization ranges.", exc_info=True)

def clone_params_with_override(base_params, override_dict=None, context=None):
    cloned_params = copy.deepcopy(base_params)
    cloned_params.normalization_ranges = {}
    cloned_params.scene_normalization_ranges = copy.deepcopy(getattr(base_params, 'scene_normalization_ranges', {}))
    if override_dict:
        for key, value in override_dict.items():
            if value is None:
                continue
            if hasattr(cloned_params, key):
                setattr(cloned_params, key, value)
    is_valid, warnings = validate_parameters(cloned_params, verbose=False)
    if warnings:
        prefix = f'[{context}] ' if context else ''
        for w in warnings:
            logger.debug("%s%s", prefix, w)
    return cloned_params

def _get_default_range(value_type, fallback_value):
    if value_type in DEFAULT_NORMALIZATION_RANGES:
        base_min, base_max = DEFAULT_NORMALIZATION_RANGES[value_type]
    else:
        span = max(1.0, abs(fallback_value))
        base_min, base_max = (0.0, span)
    if base_max - base_min < 1e-06:
        base_max = base_min + 1.0
    return [base_min, base_max]

def _update_adaptive_range(params, value_type, observed_value):
    if params is None:
        return DEFAULT_NORMALIZATION_RANGES.get(value_type, (0.0, 1.0))
    scene_ranges = getattr(params, 'scene_normalization_ranges', None)
    if scene_ranges and value_type in scene_ranges:
        return scene_ranges[value_type]
    if value_type not in params.normalization_ranges:
        params.normalization_ranges[value_type] = _get_default_range(value_type, observed_value)
    min_range, max_range = params.normalization_ranges[value_type]
    span = max(max_range - min_range, 1.0)
    guard = max(params.normalization_guard_band, 0.0)
    updated = False
    if observed_value < min_range:
        min_range = observed_value - guard * span
        updated = True
    if observed_value > max_range:
        max_range = observed_value + guard * span
        updated = True
    if max_range - min_range < 1e-06:
        max_range = min_range + 1.0
        updated = True
    if updated:
        params.normalization_ranges[value_type] = [min_range, max_range]
    return (min_range, max_range)

def normalize_value(value, value_type, iteration, min_val=None, max_val=None, params=None):
    if min_val is not None and max_val is not None:
        if max_val - min_val < 1e-09:
            return 0.0
        return (value - min_val) / (max_val - min_val)
    if params is not None:
        scene_ranges = getattr(params, 'scene_normalization_ranges', None)
        if scene_ranges and value_type in scene_ranges:
            min_range, max_range = scene_ranges[value_type]
            if max_range - min_range < 1e-09:
                return 0.0
            clipped_value = np.clip(value, min_range, max_range)
            return (clipped_value - min_range) / (max_range - min_range)
    if params is not None and getattr(params, 'normalization_mode', 'adaptive') == 'adaptive':
        min_range, max_range = _update_adaptive_range(params, value_type, value)
        span = max(max_range - min_range, 1e-09)
        clipped_value = np.clip(value, min_range, max_range)
        return (clipped_value - min_range) / span
    if value_type in DEFAULT_NORMALIZATION_RANGES:
        min_range, max_range = DEFAULT_NORMALIZATION_RANGES[value_type]
        if max_range - min_range < 1e-09:
            return 0.0
        clipped_value = np.clip(value, min_range, max_range)
        return (clipped_value - min_range) / (max_range - min_range)
    return value / (abs(value) + 1.0)
_PHYSICAL_CONFIG_ALIASES = {'pipe_diameter_m': 'pipe_diameter', 'flow_rate_m3_s': 'flow_rate', 'pipe_carbon_factor_kgco2_m': 'pipe_carbon_factor', 'elbow_carbon_factor_kgco2_each': 'elbow_carbon_factor', 'grid_carbon_intensity_kgco2_kwh': 'grid_carbon_intensity', 'fluid_density_kg_m3': 'fluid_density', 'gravity_m_s2': 'gravity', 'darcy_friction_factor': 'darcy_friction', 'pump_efficiency': 'pump_efficiency', 'annual_hours': 'annual_hours', 'reference_pipe_diameter_m': 'reference_pipe_diameter', 'pipe_carbon_scale_exponent': 'pipe_carbon_scale_exponent'}
_DEACO_CONFIG_SECTIONS = ('parameters', 'aco', 'fitness_weights', 'normalization', 'green_coefficients', 'geometry')

def _apply_if_supported(params: DEACOParameters, values: dict[str, Any], aliases: dict[str, str] | None=None) -> None:
    aliases = aliases or {}
    for key, value in values.items():
        target = aliases.get(key, key)
        if hasattr(params, target):
            setattr(params, target, value)

def _from_config(cls, config: dict[str, Any] | None=None, **overrides: Any) -> DEACOParameters:
    params = cls()
    config = config or {}
    _apply_if_supported(params, config.get('physical_parameters', {}), _PHYSICAL_CONFIG_ALIASES)
    deaco_cfg = config.get('deaco', {}) or {}
    for section in _DEACO_CONFIG_SECTIONS:
        payload = deaco_cfg.get(section, {})
        if isinstance(payload, dict):
            _apply_if_supported(params, payload)
    direct = {key: value for key, value in deaco_cfg.items() if not isinstance(value, dict)}
    _apply_if_supported(params, direct)
    _apply_if_supported(params, overrides)
    return params
DEACOParameters.from_config = classmethod(_from_config)
