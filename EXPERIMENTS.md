## Best models so far
## Scripts to launch experiments


### Elman
Script on COD dataset that got ~1.5 on val set after 500 epochs
```python PredX_train.py --data COD (--use-X)```

Script on QM9 dataset that got ~1.5 on val set after 500 epochs
```python PredX_train.py --data QM9 --mpnn-steps 3 (--use-X)```


### Seokho
Script on COD datset that got 1.520 on val set after 525 epochs
```python PredX_train.py --data COD --use-X```

Script on COD datset that got 1.546 on val set after 270 epochs
```python PredX_train.py --data COD```

Script on COD datset that got 1.714 on val set after 25 epochs
```python PredX_train.py --data COD --w-reg 1e-5```

Script on QM9 datset that got 0.529 on val set after 540 epochs
```python PredX_train.py --data QM9 --mpnn-steps 3 --w-reg 1e-5```
