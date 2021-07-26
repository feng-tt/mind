import numpy as np

np.random.seed(0)
mn = 2;sigma = 1;sampleNo = 40;
# 生成40个服从N(2,1)得随机数
Test = np.random.normal(mn,sigma,sampleNo)

# 正态分布检验 小样本3<=n<=50 夏皮洛-威尔克检验（）
import scipy.stats as stats
w,p=stats.shapiro(Test)
print(str(w)+','+str(p))
# W的值越接近1，表明数据和正态分布拟合得越好，P值>0.05，不拒绝原假设，认为样本数据服从正态分布