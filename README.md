# olfaction_evolution
Studying how glomeruli can evolve from training.


Figure 1: Standard network with a receptor layer

```python main.py --analyze receptor_standard```

Figure SX: Control for standard network without receptor layer

```python main.py --analyze control_standard```

Figure SX: Impact of pruning

```python main.py --analyze control_pn2kc_prune_boolean```

Figure SX: Impact of relabel dataset

```python main.py --analyze control_relabel_prune```

Figure SX: Training ORN-output network on relabel datasets

```python main.py --analyze control_relabel_singlelayer```

Figure SX: RNN

```python main.py --analyze rnn_relabel```

Figure SX: Vary correlation level of ORNs

```python main.py --analyze vary_orn_corr_relabel```

Figure SX: Impact of KC normalization

```python main.py --analyze kc_norm```

Figure 2: Multi-task training

```python main.py --analyze multihead_standard```

Figure 3: Meta learning

```python main.py --analyze meta_standard```

Figure SX: Meta learning controls

```python main.py --analyze meta_control_standard```

Figure 3: Scaling K-N plot

```python main.py --analyze scaling```

Figure SX: Vary number of ORs

```python main.py --analyze vary_or```

```python main.py --analyze meta_vary_or```

Figure 4: Effect of non-negativity

```python main.py --analyze control_nonnegative```

Figure 4: Vary number of KCs

```python main.py --analyze control_vary_kc```

Figure 4: Vary number of PNs

```python main.py --analyze control_vary_pn```

Figure 4: PN normalization

```python main.py --analyze pn_norm_relabel```