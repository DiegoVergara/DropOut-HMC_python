import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import seaborn as sns
import tensorflow as tf
from edward.models import Categorical, Normal, Empirical
import edward as ed
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sns.set(color_codes=True)

X_train = pd.read_csv("../data/ADIENCE/vgg_face_avg/X_train.csv", sep =",", names = None, header = None)
Y_train = pd.read_csv("../data/ADIENCE/vgg_face_avg/Y_train.csv", sep =",", names = None, header = None)
X_test = pd.read_csv("../data/ADIENCE/vgg_face_avg/X_test.csv", sep =",", names = None, header = None)
Y_test = pd.read_csv("../data/ADIENCE/vgg_face_avg/Y_test.csv", sep =",", names = None, header = None)
nb_classes = len(Y_train[0].unique())
    
path = ""
'''
lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_test = lb.transform(Y_test)
'''
start_time = time.time()
ed.set_seed(314159)
N = 100   # number of images in a minibatch.
D = X_train.shape[1]   # number of features.
num_examples = X_train.shape[0]
K = nb_classes   # number of classes.
p_samples=20
epoch = 100
num_batches = int(float(num_examples) / N)
n_samples=epoch*num_batches
friction=1.0
step_size = 1e-2
d_rate = 1.0

print "Epoch: %d, MiniBatch: %d, N Samples: %d, P Samples: %d, Friction: %.5f, StepSize: %.5f, DropOutRate: %.2f." % (epoch, N, n_samples, p_samples, friction, step_size, d_rate)

x = tf.placeholder(tf.float32, [None, D])
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
keep_prob=tf.placeholder(tf.float32)
full_drop=tf.nn.dropout(x,keep_prob=keep_prob)
y = Categorical(tf.matmul(full_drop,w)+b)

qw= Empirical(params=tf.Variable(tf.random_normal([n_samples,D,K])))
qb= Empirical(params=tf.Variable(tf.random_normal([n_samples,K])))
      
y_ph = tf.placeholder(tf.int32, [N])
#inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})
inference = ed.SGHMC({w: qw, b: qb}, data={y:y_ph})


inference.initialize(n_iter=n_samples+500, n_print=n_samples, scale={y: num_batches},step_size=step_size,friction=friction)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def next_batch(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

for k in range(epoch):
    print "Epoch: %d" %(k)
    for i,j in zip(next_batch(X_train, N),next_batch(Y_train, N)):
        X_batch, Y_batch = i, j
        X_batch = X_batch - X_train.mean(axis=0)
        info_dict = inference.update(feed_dict={x: X_batch.values, y_ph: Y_batch.values.flatten(),keep_prob:d_rate})
        inference.print_progress(info_dict)

X_test = X_test - X_train.mean(axis=0);

prob_lst = []
samples = []
w_samples = []
b_samples = []
for k in range(p_samples):
    print "Essemble: %d" %(k)
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)
    X_drop=tf.nn.dropout(tf.cast(X_test.values, tf.float32),keep_prob=d_rate)
    prob = tf.nn.softmax(tf.matmul( X_drop,w_samp ) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp,[-1]),b_samp],0)
    samples.append(sample.eval())


accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()*100
    accy_test.append(acc)

print "Elapsed time %f, seconds" % (time.time()-start_time)

prob_mean = np.mean(prob_lst,axis=0)
prob_var = np.var(prob_lst,axis=0)
prob_max = np.max(prob_mean,axis=1)


Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1)

print(classification_report(Y_test, Y_pred))
print confusion_matrix(Y_test, Y_pred)

print "accuracy in predicting the test data = %.3f :" % (Y_pred == Y_test.values).mean()*100

result = np.concatenate((prob_mean, np.reshape(prob_max,(-1,1)), np.reshape(Y_pred,(-1,1)),np.reshape(Y_test,(-1,1)),prob_var),axis=1)
np.savetxt(path+"SGHMC_D_age_analysis.csv", result, fmt="%1.3f", header ="mean_0, mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, max_prob, pred, GT, var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7",delimiter = ",")


#sns.distplot(accy_test)
plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the AGE test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.savefig(path+"SGDHMC_D_AGE_t_data_acc_freq.png")
#plt.show()
plt.close()

'''
samples_df = pd.DataFrame(data = samples, index=range(p_samples))
samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values,columns=["W_0", "W_1", "W_2", "W_3", "W_4"])
g = sns.PairGrid(samples_5, diag_sharey=False)
g.map_lower(sns.kdeplot, n_levels = 4,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,legend=False)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Joint posterior distribution of the first 5 weights')
plt.savefig("MNIST_first_5_w.png")
plt.show()
plt.close()

test_image = X_test[4:5]
test_label = Y_test[4]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("MNIST_gt.png")
plt.show()
plt.close()


sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul( X_test[4:5],w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Accuracy of the prediction of the test digit")
plt.ylabel("Frequency")
plt.savefig("MNIST_digit_acc_freq.png")
plt.show()
plt.close()
'''



