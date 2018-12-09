## Best models so far
## Scripts to launch experiments


### Elman
Script on COD dataset that got ~1.5 on val set after 500 epochs
```python PredX_train.py --data COD (--use-X)```

Script on QM9 dataset that got ~1.5 on val set after 500 epochs
```python PredX_train.py --data QM9 --mpnn-steps 3 (--use-X)```


### Seokho
Script on COD datset that got 1.468 on val set after 1000 epochs
```python PredX_train.py --data COD --w-reg 1e-5```

Script on QM9 datset that got 0.467 on val set after 2500 epochs
```python PredX_train.py --data QM9 --mpnn-steps 3 --w-reg 1e-5```
