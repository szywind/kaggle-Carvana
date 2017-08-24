
class RMSpropAccum(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, decay=0., accumulator=5.,
                 **kwargs):
        super(RMSpropAccum, self).__init__(**kwargs)
        self.lr = K.variable(lr, name='lr')
        self.rho = K.variable(rho, name='rho')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.iterations = K.variable(0., name='iterations')
        self.accumulator = K.variable(accumulator, name='accumulator')

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        gs =           [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        for p, g, a, ga in zip(params, grads, accumulators, gs):

            flag = K.equal(self.iterations % self.accumulator, 0)
            flag = K.cast(flag, dtype='float32')

            ga_t = (1 - flag) * (ga + g)

            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(ga + flag * g) / self.accumulator
            p_t = p - lr * (ga + flag * g) / self.accumulator / (K.sqrt(new_a) + self.epsilon)

            self.updates.append(K.update(a, flag * new_a + (1 - flag) * a))
            self.updates.append(K.update(ga, ga_t))

            new_p = flag * p_t + (1 - flag) * p

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'accumulator': float(K.get_value(self.accumulator)),
                  'epsilon': self.epsilon}
        base_config = super(RMSpropAccum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
