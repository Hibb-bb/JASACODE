"""
Test script to verify mixed graph structure implementation.
"""
import torch
import numpy as np

from data import (
    get_mixed_graph_structures,
    get_structure_names,
    compile_template_from_structure,
    init_graph_params_uniform,
    MixedICLBatchSpec,
    MixedGraphICLSequenceDataset,
)


def test_graph_structures():
    """Test that all structures have 5 nodes."""
    print("Testing graph structures...")
    structures = get_mixed_graph_structures(seed=42)
    names = get_structure_names()
    
    assert len(structures) == 3, "Should have 3 structures"
    assert len(names) == 3, "Should have 3 names"
    
    for i, (bn, name) in enumerate(zip(structures, names)):
        print(f"  Structure {i} ({name}): {len(bn.nodes)} nodes")
        assert len(bn.nodes) == 5, f"Structure {name} should have 5 nodes"
        print(f"    Nodes: {bn.nodes}")
        
        # Count edges by summing parent relationships
        num_edges = sum(len(bn._parents.get(node, [])) for node in bn.nodes)
        print(f"    Edges: {num_edges}")
    
    print("✓ Graph structures test passed\n")


def test_templates():
    """Test template compilation."""
    print("Testing template compilation...")
    structures = get_mixed_graph_structures(seed=42)
    templates = [compile_template_from_structure(bn) for bn in structures]
    
    for i, template in enumerate(templates):
        print(f"  Template {i}:")
        print(f"    Nodes: {template.num_nodes}")
        print(f"    Topo order: {template.topo_nodes}")
        assert template.num_nodes == 5, f"Template {i} should have 5 nodes"
    
    print("✓ Template compilation test passed\n")


def test_mixed_dataset():
    """Test mixed dataset creation and batching."""
    print("Testing mixed dataset...")
    
    # Setup
    structures = get_mixed_graph_structures(seed=42)
    templates = [compile_template_from_structure(bn) for bn in structures]
    
    # Initialize parameters (small for testing)
    p1_lists = [
        init_graph_params_uniform(template, num_graphs=10, seed=i)
        for i, template in enumerate(templates)
    ]
    
    # Create spec
    spec = MixedICLBatchSpec(
        batch_graphs=12,  # Divisible by 3
        num_example=5,
        target_index=4,
    )
    
    # Create dataset
    dataset = MixedGraphICLSequenceDataset(
        templates=templates,
        p1_lists=p1_lists,
        structure_names=get_structure_names(),
        seed=42,
        spec=spec,
    )
    
    # Get a batch
    batch = next(iter(dataset))
    
    # Check shapes
    B = 12
    L = 6  # num_example + 1
    N = 5
    
    print(f"  Batch shapes:")
    print(f"    x: {batch['x'].shape} (expected: ({B}, {L}, {N+1}))")
    print(f"    y: {batch['y'].shape} (expected: ({B},))")
    print(f"    graph_id: {batch['graph_id'].shape} (expected: ({B},))")
    print(f"    structure_id: {batch['structure_id'].shape} (expected: ({B},))")
    
    assert batch['x'].shape == (B, L, N+1), f"Wrong x shape: {batch['x'].shape}"
    assert batch['y'].shape == (B,), f"Wrong y shape: {batch['y'].shape}"
    assert batch['graph_id'].shape == (B,), f"Wrong graph_id shape"
    assert batch['structure_id'].shape == (B,), f"Wrong structure_id shape"
    
    # Check structure distribution
    structure_counts = torch.bincount(batch['structure_id'])
    print(f"  Structure distribution: {structure_counts.tolist()}")
    print(f"    Expected: ~{B//3} per structure")
    
    # Check data validity
    x_values = batch['x'][:, :, :-1]  # Exclude target index
    print(f"  Data range: [{x_values.min().item()}, {x_values.max().item()}] (expected: [0, 1])")
    assert x_values.min() >= 0 and x_values.max() <= 1, "Values should be binary"
    
    # Check target index
    target_indices = batch['x'][:, :, -1]
    print(f"  Target indices: unique values = {target_indices.unique().tolist()} (expected: [4])")
    assert (target_indices == 4).all(), "All target indices should be 4"
    
    # Check masking
    test_row = batch['x'][:, -1, :N]  # Last row, exclude target index
    print(f"  Test row masking: columns {spec.target_index} onward should be 0")
    assert (test_row[:, spec.target_index:] == 0).all(), "Test row should mask target and future"
    
    print("✓ Mixed dataset test passed\n")


def test_batch_distribution():
    """Test that structures are distributed evenly across batches."""
    print("Testing batch distribution across structures...")
    
    structures = get_mixed_graph_structures(seed=42)
    templates = [compile_template_from_structure(bn) for bn in structures]
    p1_lists = [
        init_graph_params_uniform(template, num_graphs=100, seed=i)
        for i, template in enumerate(templates)
    ]
    
    spec = MixedICLBatchSpec(
        batch_graphs=60,  # Divisible by 3
        num_example=10,
        target_index=4,
    )
    
    dataset = MixedGraphICLSequenceDataset(
        templates=templates,
        p1_lists=p1_lists,
        structure_names=get_structure_names(),
        seed=42,
        spec=spec,
    )
    
    # Check multiple batches
    structure_counts_total = torch.zeros(3, dtype=torch.long)
    num_batches = 10
    
    for i, batch in enumerate(dataset):
        if i >= num_batches:
            break
        structure_counts = torch.bincount(batch['structure_id'], minlength=3)
        structure_counts_total += structure_counts
    
    print(f"  Total counts over {num_batches} batches:")
    print(f"    Structure 0 (tree): {structure_counts_total[0].item()}")
    print(f"    Structure 1 (chain): {structure_counts_total[1].item()}")
    print(f"    Structure 2 (general): {structure_counts_total[2].item()}")
    print(f"    Expected: ~{60 * num_batches // 3} each")
    
    # Check they're roughly equal (within 10%)
    expected_per_structure = 60 * num_batches / 3
    for i in range(3):
        count = structure_counts_total[i].item()
        diff_pct = abs(count - expected_per_structure) / expected_per_structure * 100
        print(f"    Structure {i} deviation: {diff_pct:.1f}%")
        assert diff_pct < 10, f"Structure {i} count too far from expected"
    
    print("✓ Batch distribution test passed\n")


def test_uneven_batch_size():
    """Test batch size not divisible by 3."""
    print("Testing uneven batch size...")
    
    structures = get_mixed_graph_structures(seed=42)
    templates = [compile_template_from_structure(bn) for bn in structures]
    p1_lists = [
        init_graph_params_uniform(template, num_graphs=10, seed=i)
        for i, template in enumerate(templates)
    ]
    
    # Batch size = 10 (not divisible by 3)
    spec = MixedICLBatchSpec(
        batch_graphs=10,
        num_example=5,
        target_index=4,
    )
    
    dataset = MixedGraphICLSequenceDataset(
        templates=templates,
        p1_lists=p1_lists,
        structure_names=get_structure_names(),
        seed=42,
        spec=spec,
    )
    
    batch = next(iter(dataset))
    structure_counts = torch.bincount(batch['structure_id'])
    print(f"  Batch size 10, structure distribution: {structure_counts.tolist()}")
    print(f"    Expected: [3, 3, 4] or similar")
    
    assert structure_counts.sum() == 10, "Total should be 10"
    assert len(structure_counts) == 3, "Should have 3 structures"
    
    print("✓ Uneven batch size test passed\n")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Mixed Graph Structure Implementation")
    print("="*60 + "\n")
    
    try:
        test_graph_structures()
        test_templates()
        test_mixed_dataset()
        test_batch_distribution()
        test_uneven_batch_size()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
