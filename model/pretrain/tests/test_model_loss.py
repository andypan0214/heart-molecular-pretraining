"""Stage B.3-B.7 -- random-init model forward, LUMI multi-objective loss, backward,
gradient flow and checkpoint serialisation (CPU, tiny architecture)."""
import copy

import pytest
import torch

LOSS_TERMS = ["masked_token_loss", "masked_coord_loss", "masked_dist_loss",
              "x_norm_loss", "delta_pair_repr_norm_loss", "contrastive_loss"]


def build(task, seed=0):
    from unimol.models.unimol_contrastive import UniMolModelContrastive
    from unimol.losses.unimol_contrastive import UniMolLossContrastive

    torch.manual_seed(seed)
    model = UniMolModelContrastive(copy.deepcopy(task.args), task.dictionary).train()
    return model, UniMolLossContrastive(task)


@pytest.fixture
def setup(make_task):
    task, ds = make_task()
    batch = ds.collater([ds[i] for i in range(4)])
    model, loss_fn = build(task)
    return task, batch, model, loss_fn


def test_forward_output_shapes(setup):
    task, batch, model, _ = setup
    masked = batch["target"]["tokens_target"].ne(task.dictionary.pad())
    logits, dist, coord, x_norm, pair_norm, emb = model(**batch["net_input"], encoder_masked_tokens=masked)
    n, L = batch["net_input"]["src_tokens"].shape
    assert logits.shape == (int(masked.sum()), len(task.dictionary))
    assert dist.shape == (n, L, L) and coord.shape == (n, L, 3)
    assert emb.shape == (n, task.args.encoder_embed_dim)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(n), atol=1e-5)  # L2-normalised for NT-Xent
    assert x_norm.ndim == 0 and pair_norm.ndim == 0


def test_all_lumi_objectives_finite(setup):
    _, batch, model, loss_fn = setup
    loss, sample_size, logs = loss_fn(model, batch)
    assert torch.isfinite(loss) and sample_size == 1
    for k in LOSS_TERMS:
        assert k in logs, "objective {} not computed".format(k)
        assert torch.isfinite(torch.as_tensor(logs[k])), k
    a = loss_fn.args
    expected = (logs["masked_token_loss"] * a.masked_token_loss
                + logs["masked_coord_loss"] * a.masked_coord_loss
                + logs["masked_dist_loss"] * a.masked_dist_loss
                + logs["x_norm_loss"] * a.x_norm_loss
                + logs["delta_pair_repr_norm_loss"] * a.delta_pair_repr_norm_loss
                + logs["contrastive_loss"] * a.contrastive_loss)
    assert torch.allclose(loss.detach(), torch.as_tensor(expected), rtol=1e-5)


def test_backward_gradients_reach_every_head(setup):
    _, batch, model, loss_fn = setup
    loss, _, _ = loss_fn(model, batch)
    loss.backward()
    groups = ["embed_tokens", "encoder.layers.0", "encoder.layers.1", "gbf.", "gbf_proj",
              "lm_head", "pair2coord_proj", "dist_head", "proj_head"]
    for g in groups:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(g)]
        assert grads and all(x is not None for x in grads), g
        assert all(torch.isfinite(x).all() for x in grads), g
        assert any(x.abs().sum() > 0 for x in grads), "zero gradient in " + g


def test_contrastive_objective_alone_trains_projection_head(setup):
    """proj_head is used only by the NT-Xent term: proves the contrastive path is wired."""
    _, batch, model, loss_fn = setup
    for k in ("masked_token_loss", "masked_coord_loss", "masked_dist_loss",
              "x_norm_loss", "delta_pair_repr_norm_loss"):
        setattr(loss_fn.args, k, 0.0)
    loss, _, logs = loss_fn(model, batch)
    assert torch.allclose(loss.detach(), logs["contrastive_loss"] * loss_fn.args.contrastive_loss)
    loss.backward()
    assert model.proj_head[0].weight.grad.abs().sum() > 0


def test_optimizer_step_updates_and_loss_decreases(setup):
    _, batch, model, loss_fn = setup
    model.eval()  # remove dropout noise so the comparison is meaningful
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = loss_fn(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert len(changed) > 0.8 * len(before)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]  # overfits one fixed batch: plumbing check, not learning


def test_no_padding_batch_does_not_crash(make_task):
    """HEART patch P4: upstream dereferenced padding_mask=None in the coord head."""
    task, _ = make_task()
    model, loss_fn = build(task)
    d = task.dictionary
    L = 8  # every sequence full length -> no [PAD] anywhere
    heavy = torch.tensor([d.index(s) for s in ["C", "C", "O", "N", "C", "C"]])
    seq = torch.cat([torch.tensor([d.bos()]), heavy, torch.tensor([d.eos()])])
    tok = seq.repeat(4, 1)
    tok[:, 2] = task.mask_idx
    coord = torch.randn(4, L, 3)
    coord[:, 0] = coord[:, -1] = 0
    dist = torch.cdist(coord, coord)
    V = len(d)
    tgt = torch.full((4, L), d.pad())
    tgt[:, 2] = d.index("C")
    batch = {"net_input": {"src_tokens": tok, "src_coord": coord, "src_distance": dist,
                           "src_edge_type": tok.unsqueeze(-1) * V + tok.unsqueeze(1)},
             "target": {"tokens_target": tgt, "distance_target": dist, "coord_target": coord,
                        "smi_name": ["a", "b", "a", "b"]}}
    assert not tok.eq(d.pad()).any()
    loss, _, _ = loss_fn(model, batch)
    assert torch.isfinite(loss)


def test_random_initialisation(make_task):
    task, _ = make_task()
    m1, _ = build(task, seed=1)
    m2, _ = build(task, seed=2)
    w1, w2 = m1.state_dict(), m2.state_dict()
    assert not torch.equal(w1["embed_tokens.weight"], w2["embed_tokens.weight"])
    assert not torch.equal(w1["proj_head.0.weight"], w2["proj_head.0.weight"])


def test_checkpoint_roundtrip(setup, tmp_path):
    _, batch, model, loss_fn = setup
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss, _, _ = loss_fn(model, batch)
    loss.backward()
    opt.step()
    path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "num_updates": 1}, path)

    task = loss_fn.task
    model2, _ = build(task, seed=123)
    ck = torch.load(path, map_location="cpu")
    model2.load_state_dict(ck["model"], strict=True)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    opt2.load_state_dict(ck["optimizer"])
    assert ck["num_updates"] == 1
    model.eval(), model2.eval()
    with torch.no_grad():
        l1, _, _ = loss_fn(model, batch)
        l2, _, _ = loss_fn(model2, batch)
    assert torch.equal(l1, l2)


def test_lumi_base_architecture_builds_and_runs(make_task):
    """Full LUMI base size (15 layers, 512 dim, 64 heads) -- the pilot's architecture."""
    from unimol.models.unimol_contrastive import UniMolModelContrastive

    task, ds = make_task()
    args = copy.deepcopy(task.args)
    for k in ("encoder_layers", "encoder_embed_dim", "encoder_ffn_embed_dim"):
        delattr(args, k)  # fall back to the registered architecture defaults
    args.encoder_attention_heads = 64
    torch.manual_seed(0)
    model = UniMolModelContrastive(args, task.dictionary)
    assert (args.encoder_layers, args.encoder_embed_dim, args.encoder_ffn_embed_dim) == (15, 512, 2048)
    n_params = sum(p.numel() for p in model.parameters())
    print("LUMI base parameters:", n_params)
    assert 40e6 < n_params < 60e6
    batch = ds.collater([ds[0], ds[1]])
    masked = batch["target"]["tokens_target"].ne(task.dictionary.pad())
    with torch.no_grad():
        out = model(**batch["net_input"], encoder_masked_tokens=masked)
    assert all(torch.isfinite(t).all() for t in out)
