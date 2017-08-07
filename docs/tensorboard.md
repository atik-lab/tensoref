# Tensorboard

```python
writer = tf.summary.FileWriter('./graphs', sess.graph)`
writer.close()

tensorboard --logdir="/home/pere/Projects/tensoref/data/"
```

Also interesting for debugging:

```python
print(tf.get_default_graph().as_graph_def())
```