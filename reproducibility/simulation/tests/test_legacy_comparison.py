import pandas as pd

from compare_legacy_results import aggregate_legacy, aggregate_repaired


def test_legacy_comparison_maps_adaptive_to_query_type_oracle():
    legacy = pd.DataFrame(
        {
            "config_k": [3, 3],
            "type": ["semantic", "multi-hop"],
            "lru_sim_hit": [0, 1],
            "cos_sim_hit": [1, 1],
            "grp_sim_hit": [0, 1],
            "adp_sim_hit": [1, 0],
        }
    )
    aggregated = aggregate_legacy(legacy)
    adaptive = aggregated[
        aggregated.legacy_policy.eq("legacy_query_type_adaptive")
        & aggregated.query_type.eq("all")
    ].iloc[0]
    assert adaptive.repaired_policy == "adaptive_offline_query_type_oracle"
    assert adaptive.legacy_residency_like_rate == 0.5

    repaired = pd.DataFrame(
        {
            "k": [3],
            "policy": ["graph_fixed"],
            "query_type": ["semantic"],
            "prediction_hit": [True],
            "residency_hit": [True],
            "current_prefetch_hit": [False],
        }
    )
    assert aggregate_repaired(repaired).query_type.tolist() == ["all", "semantic"]
