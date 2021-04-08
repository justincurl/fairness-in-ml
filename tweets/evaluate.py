from __future__ import division
import langid
import sys, re, os, random, itertools
import numpy as np 
import HTMLParser
import twokenize
import predict
import emoji
import langid
import ujson as json

HTML_PARSER = HTMLParser.HTMLParser()

def process_tweet(tweet):

    """ Fix HTML, remove emojis, remove @-mentions, remove hashtags. """

    tweet = tweet.decode('utf-8')
    tweet = tweet.replace("&amp;", "&")
    tweet = HTML_PARSER.unescape(tweet)
    # tweet = emoji.clean_emoji_and_symbols(tweet)
    # tweet = re.sub("@[A-Za-z0-9_]+\s*", "", tweet)
    # tweet = re.sub("RT", "", tweet)
    # tweet = re.sub('(http)[\w\/:.]+\s*', "", tweet)
    toks = twokenize.tokenizeRawTweetText(tweet)

    return tweet, toks

def predict_lang(toks, alpha=.125, numpasses=5, thresh1=1, thresh2=0.1, thresh3=0.1):

    final_lang = ''
    posterior = predict.predict(toks, alpha, numpasses, thresh1, thresh2)
    if posterior is not None:
        if posterior[2] < thresh3:
            final_lang = 'en'
    return final_lang

""" Experiments. """

messages = []
infile = 'all_annotated.tsv'

for line in open(infile):

	L = line.split("\t")
	# ignore header
	if L[0].startswith("T"): continue
	tweet, toks = process_tweet(L[3])

	# ignore automatically generated and ambiguous tweets
	auto = int(L[-1])
	if auto == 1: continue
	amb = int(L[5])
	if amb == 1: continue

	en = int(L[4])
	nonen = int(L[6])
	lang = ""
	if en == 1:
		lang = "en"
	else:
		assert nonen == 1
		lang = "nonen"

	messages.append((tweet, toks, lang))

""" Testing. """

conf_matrix = np.zeros([2,2])

predict.load_model('model_count_table')

for (tweet, toks, lang) in messages:

	langid_lang = langid.classify(tweet)[0]
	dem_lang = predict_lang(toks)

	if langid_lang == "en" or dem_lang == "en":
		final_lang = "en"
	else:
		final_lang = "nonen"

	if ann_lang == "en" and final_lang == "en":
		conf_matrix[1][1] += 1
	elif ann_lang == "en" and final_lang != "en":
		conf_matrix[1][0] += 1
	elif ann_lang != "en" and final_lang == "en":
		conf_matrix[0][1] += 1
	else:
		conf_matrix[0][0] += 1

prec = conf_matrix[1][1] / np.sum(conf_matrix[:,1])
rec = conf_matrix[1][1] / np.sum(conf_matrix[1])

print "Classifier: precision %s, recall %s" % (prec, rec)