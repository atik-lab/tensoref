# Global step variable

## Learning rate can be a tensor!

```python
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
increment_step = global_step.assign_add(1)

optimizer = tf.GradientDescentOptimizer(learning_rate)
```