## machine learning 误差的来源

线性模型，有两个误差：bias 和 variance

### 1. bias 大

* 重新设计/写 model，增加次数，变复杂
* 考虑更多的因素，以 pm2.5 为例，把其他一些变量考虑进去
* 去找更多数据，是没有用的！

### 2. variance 大

* 收集更多 data （万能方法，但是收集 data 比较难）
* 根据自己的理解，自己制造 data(影像识别，左右颠倒。语音标识，变声器变音)
* 函式加上 Regularization，得到更平滑的曲线，但是可能会伤害bias

### 不应该做什么？

在机器学习中，可能有 training data、testing data 两种数据。

前者用于训练，后者用于测试。

当我们想要知道哪种model更好的时候，单纯用training data训练出结果apply到testing data上看error是不可取的。因为我们的函式以后应用在实际生活中，接收到的数据是未知的，如果去根据testing data调整，相当于是把testing data也考虑进去了，最后得到的结果可能比预期的差。

**那么应该怎么做才能知道哪个model更好呢？**

可以把 training data 分成 traing data 和 validation data，然后用 training data 测出来的 apply 到 validation data 上，也就是所谓的用验证集去验证结果。

还可以把training data 分成三份，取其中一份作为validation data，然后依次做三次，取最后得到error的平均数，最低的那个可以认为是更好的 model。

**那把 training data这样划分，用来训练的training data会不会太少了呢？**

如果有这方面考虑，那么得到更好的model后，可以再重新用全部training data算一遍。

理论上，如果不对testing data做调整，最后的结果误差不会太多。