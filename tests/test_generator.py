from cp_toy.config import DataConfig
from cp_toy.data import ChainBatchGenerator


def test_shapes_and_mask():
    cfg = DataConfig(v_content=64, chain_length=8, k_max=2, sep_token=64, query_a_token=65, query_b_token=66, query_mem_token=67, hop_token_offset=68)
    gen = ChainBatchGenerator(cfg, seed=0)
    b = gen.batch(4, p_dynamic=1.0, p_multi=0.5)
    assert b.input_ids.shape == (4, cfg.input_seq_len)
    assert b.labels.shape == (4, cfg.input_seq_len)
    assert b.loss_mask.sum().item() == 4
    assert b.loss_mask[:, -1].sum().item() == 4


def test_keyslot_positions_are_in_input():
    cfg = DataConfig(v_content=64, chain_length=8, k_max=2, sep_token=64, query_a_token=65, query_b_token=66, query_mem_token=67, hop_token_offset=68)
    gen = ChainBatchGenerator(cfg, seed=1)
    b = gen.diagnostic_keyslot_batch(16)
    assert (b.key_value_pos >= 0).all()
    assert (b.key_value_pos < cfg.input_seq_len).all()
    assert (b.query_pos == cfg.query_start_input_pos).all()
    assert (b.hop == 1).all()
