import tensorflow as tf

BATCH_SIZE = 128
VOCAB_SIZE = 32
EMBED_SIZE = 1024
LOGDIR = '/tmp/w2v/'

with tf.name_scope('data'):
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

with tf.device('/gpu:0'):
    with tf.name_scope("embed"):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),
                                   name='embed_matrix')
    # Step 3 + 4: define the inference + the loss function
    with tf.name_scope("loss"):
        # Step 3: define the inference
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        # Step 4: construct variables for NCE loss
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], name='nce_weight'))
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                             biases=nce_bias, labels=target_words,
                                             inputs=embed,
                                             num_sampled=100,
                                             num_classes=VOCAB_SIZE), name='loss')
        # Step 5: define optimizer
        optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

from tensorflow.contrib.tensorboard.plugins import projector

sess = tf.InteractiveSession()
# obtain the embedding_matrix after you’ve trained it
final_embed_matrix = sess.run(model.embed_matrix)
# create a variable to hold your embeddings. It has to be a variable. Constants
# don’t work. You also can’t just use the embed_matrix we defined earlier for our model. Why
# is that so? I don’t know. I get the 500 most popular words.
embedding_var = tf.Variable(final_embed_matrix[:500], name='embedding')
sess.run(embedding_var.initializer)
config = projector.ProjectorConfig()
summary_writer = tf.summary.FileWriter(LOGDIR)
# add embeddings to config
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# link the embeddings to their metadata file. In this case, the file that contains
# the 500 most popular words in our vocabulary
embedding.metadata_path = LOGDIR + '/vocab_500.tsv'
# save a configuration file that TensorBoard will read during startup
projector.visualize_embeddings(summary_writer, config)
# save our embedding
saver_embed = tf.train.Saver([embedding_var])
saver_embed.save(sess, LOGDIR + '/skip-gram.ckpt', 1)
