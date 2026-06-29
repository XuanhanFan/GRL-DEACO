"""Shared metric collection utilities for GRL-DEACO evaluation scripts."""

from typing import Dict, Optional

import numpy as np


class MetricsCollector:
    """Collect per-connection routing metrics and summarize them."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.path_lengths = []
        self.bend_counts = []
        self.altitudes = []
        self.violations = []
        self.violation_counts = []

        self.energy_operational = []
        self.carbon_embedded = []
        self.carbon_operational = []

        self.success_flags = []
        self.fitness_values = []
        self.rewards = []
        self.connection_times = []
        self.total_times = []
        self.connection_details = []

    def add_connection_result(
        self,
        connection_idx: int,
        success: bool,
        fitness_data: Optional[Dict] = None,
        path_length: int = 0,
        processing_time: float = 0.0,
        reward: float = 0.0,
        fitness: Optional[float] = None,
        scenario_name: str = "",
        connection_info: Optional[Dict] = None,
        action_params: Optional[Dict] = None,
        fitness_breakdown: Optional[Dict] = None,
    ):
        self.success_flags.append(success)
        self.path_lengths.append(path_length)
        self.connection_times.append(processing_time)
        self.rewards.append(reward)

        if fitness is not None and fitness != float("inf"):
            self.fitness_values.append(fitness)

        if fitness_data is not None:
            self.energy_operational.append(fitness_data.get("E_op", 0.0))
            self.carbon_embedded.append(fitness_data.get("CO2_emb", 0.0))
            self.carbon_operational.append(fitness_data.get("CO2_op", 0.0))
            self.bend_counts.append(fitness_data.get("N_bend", 0))
            self.altitudes.append(fitness_data.get("alt", 0.0))
            self.violations.append(fitness_data.get("viol", 0.0))
            if "L" in fitness_data and fitness_data["L"] > 0:
                self.path_lengths[-1] = fitness_data["L"]
        else:
            self.energy_operational.append(0.0)
            self.carbon_embedded.append(0.0)
            self.carbon_operational.append(0.0)
            self.bend_counts.append(0)
            self.altitudes.append(0.0)
            self.violations.append(0.0)

        self.connection_details.append(
            {
                "connection_idx": connection_idx,
                "scenario": scenario_name,
                "success": success,
                "path_length": path_length,
                "processing_time": processing_time,
                "reward": reward,
                "fitness": fitness if fitness != float("inf") else None,
                "fitness_data": fitness_data,
                "connection_info": connection_info,
                "action_params": action_params,
                "fitness_breakdown": fitness_breakdown,
            }
        )

    def add_connection(
        self,
        path_length: int,
        bend_count: int,
        height_change: float,
        violation: float,
        energy_op: float,
        carbon_emb: float,
        carbon_op: float,
        success: bool,
        connection_time: float,
    ):
        self.add_connection_result(
            connection_idx=len(self.success_flags),
            success=success,
            fitness_data={
                "L": path_length,
                "N_bend": bend_count,
                "alt": height_change,
                "viol": violation,
                "E_op": energy_op,
                "CO2_emb": carbon_emb,
                "CO2_op": carbon_op,
            },
            path_length=path_length,
            processing_time=connection_time,
        )

    @staticmethod
    def _valid(values, success_flags):
        return [value for idx, value in enumerate(values) if idx < len(success_flags) and success_flags[idx]]

    @staticmethod
    def _summary(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def compute_statistics(self) -> Dict:
        total_connections = len(self.success_flags)
        successful_connections = sum(self.success_flags)
        failed_connections = total_connections - successful_connections

        valid_lengths = self._valid(self.path_lengths, self.success_flags)
        valid_bends = self._valid(self.bend_counts, self.success_flags)
        valid_alts = self._valid(self.altitudes, self.success_flags)
        valid_violations = self._valid(self.violations, self.success_flags)
        valid_energy = self._valid(self.energy_operational, self.success_flags)
        valid_embodied = self._valid(self.carbon_embedded, self.success_flags)
        valid_operational = self._valid(self.carbon_operational, self.success_flags)
        valid_times = self._valid(self.connection_times, self.success_flags)

        length_summary = self._summary(valid_lengths)
        bend_summary = self._summary(valid_bends)
        alt_summary = self._summary(valid_alts)
        energy_summary = self._summary(valid_energy)
        embodied_summary = self._summary(valid_embodied)
        operational_summary = self._summary(valid_operational)

        violation_count = sum(1 for value in valid_violations if value > 1e-6)
        violation_ratio = violation_count / len(valid_violations) if valid_violations else 0.0

        return {
            "path_quality": {
                "L_mean": length_summary["mean"],
                "L_std": length_summary["std"],
                "L_min": length_summary["min"],
                "L_max": length_summary["max"],
                "L_median": float(np.median(valid_lengths)) if valid_lengths else 0.0,
                "N_bend_mean": bend_summary["mean"],
                "N_bend_std": bend_summary["std"],
                "N_bend_min": bend_summary["min"],
                "N_bend_max": bend_summary["max"],
                "H_alt_mean": alt_summary["mean"],
                "H_alt_std": alt_summary["std"],
                "H_alt_min": alt_summary["min"],
                "H_alt_max": alt_summary["max"],
                "N_viol_mean": float(np.mean(valid_violations)) if valid_violations else 0.0,
                "N_viol_std": float(np.std(valid_violations)) if valid_violations else 0.0,
                "N_viol_count": violation_count,
                "N_viol_ratio": violation_ratio,
            },
            "green_metrics": {
                "E_op_mean": energy_summary["mean"],
                "E_op_std": energy_summary["std"],
                "E_op_min": energy_summary["min"],
                "E_op_max": energy_summary["max"],
                "E_op_total": float(np.sum(valid_energy)) if valid_energy else 0.0,
                "CO2_emb_mean": embodied_summary["mean"],
                "CO2_emb_std": embodied_summary["std"],
                "CO2_emb_min": embodied_summary["min"],
                "CO2_emb_max": embodied_summary["max"],
                "CO2_emb_total": float(np.sum(valid_embodied)) if valid_embodied else 0.0,
                "CO2_op_mean": operational_summary["mean"],
                "CO2_op_std": operational_summary["std"],
                "CO2_op_total": float(np.sum(valid_operational)) if valid_operational else 0.0,
            },
            "algorithm_performance": {
                "total_connections": total_connections,
                "successful_connections": successful_connections,
                "failed_connections": failed_connections,
                "success_rate": successful_connections / total_connections if total_connections else 0.0,
                "failure_rate": failed_connections / total_connections if total_connections else 1.0,
                "avg_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
                "avg_fitness": float(np.mean(self.fitness_values)) if self.fitness_values else float("inf"),
                "avg_time_per_connection": float(np.mean(valid_times)) if valid_times else 0.0,
                "total_time": float(np.sum(self.connection_times)) if self.connection_times else 0.0,
                "min_time": float(np.min(valid_times)) if valid_times else 0.0,
                "max_time": float(np.max(valid_times)) if valid_times else 0.0,
            },
        }

    def merge(self, other: "MetricsCollector"):
        if not isinstance(other, MetricsCollector):
            raise TypeError(f"Expected MetricsCollector, got {type(other)}")

        self.path_lengths.extend(other.path_lengths)
        self.bend_counts.extend(other.bend_counts)
        self.altitudes.extend(other.altitudes)
        self.violations.extend(other.violations)
        self.violation_counts.extend(other.violation_counts)
        self.energy_operational.extend(other.energy_operational)
        self.carbon_embedded.extend(other.carbon_embedded)
        self.carbon_operational.extend(other.carbon_operational)
        self.success_flags.extend(other.success_flags)
        self.fitness_values.extend(other.fitness_values)
        self.rewards.extend(other.rewards)
        self.connection_times.extend(other.connection_times)
        self.total_times.extend(other.total_times)
        self.connection_details.extend(other.connection_details)
