import cPickle as pickle
import re
import sys
import numpy as np
from multiprocessing import Pool
import argparse
from scipy.spatial.distance import cosine
from collections import OrderedDict
import random

#Parser 
#parser = argparse.ArgumentParser()
#parser.add_argument("-ie", "-ie")
#ie = str(parser.parse_args().ie)

ie = '../claims_codes_hs_300.txt'
threads = 1

#Loading the code mappings
icd9_to_translation_hdr = 'icd9Tree.txt'
ndc_to_translation_hdr = 'ndc_labels.txt'
loinc_to_translation_hdr = 'loinc_code_names.txt'
cpt_to_translation_hdr = 'cpt_code_names.txt'

code_to_translation_map = {}
f = open(icd9_to_translation_hdr,'r')
lines = f.readlines()
for line in lines:
    tokens = line.split(" ")
    try:
        code = "IDX_" + re.search('\((.+?)\)', tokens[0]).group(1)
        translation  = ' '.join(tokens[1:])[:-1]
        code_to_translation_map[code] = translation.title()
    except:
        continue
f = open(ndc_to_translation_hdr,'r')
lines = f.readlines()
for line in lines:
    tokens = line.split("---")
    code = tokens[0]
    translation = tokens[1][:-1]
    code = "N_" + code
    code_to_translation_map[code] = translation.title()
f = open(loinc_to_translation_hdr,'r')
lines = f.readlines()
for line in lines:
    tokens = line.split("#")
    code = "L_" + tokens[0]
    translation = tokens[1][:-1]
    code_to_translation_map[code] = translation.title()
f = open(cpt_to_translation_hdr,'r')
lines = f.readlines()
for line in lines:
    tokens = line.split("#")
    codes = tokens[0].split(" ")
    translation = ' '.join(tokens[1].split(" ")[1:-1])
    print translation
    for code in codes:
        code = 'C_' + code
        code_to_translation_map[code] = translation.title()


#Create embedding matrix and index_to_concept concept_to_index maps
f = open(ie,'r')
lines = f.readlines() 
embedding_matrix = np.zeros(shape=(len(lines)-2,len(lines[2].split(" "))-2))
print embedding_matrix.shape 
idx_to_code_map = {}
code_to_idx_map = {}
for i in xrange(2,len(lines)):
    line = lines[i]
    idx = i - 2
    key = line.split(" ")[0]
    embedding = np.array(line.split(" ")[1:-1])
    embedding_matrix[idx,:] = embedding
    idx_to_code_map[idx] = key
    code_to_idx_map[key] = idx

def compute_cosine_distance(idx_pair):
    first_idx = idx_pair[0]
    second_idx = idx_pair[1]
    return 1-cosine(embedding_matrix[second_idx,:],embedding_matrix[first_idx,:])

def compute_cosine_distance_with_vector(idx_vector_pair):
    first_idx = idx_vector_pair[0]
    vector = idx_vector_pair[1]
    return 1 - cosine(vector,embedding_matrix[first_idx,:])

def code_to_translation(code):
    if code in code_to_translation_map:
        return code_to_translation_map[code]
    else:
        return "NOT FOUND IN THE MAPPING"
        
if __name__ == "__main__":
    while True:
        try:
            input = str(raw_input('Enter analysis type & the codes (exit to break): '))
            analysis_type = input.split(" ")[0]
            codes = input.split(" ")[1:]
        except:
            input = ''
        if 'exit' in input:
            break
        try:
            if analysis_type == 'examples':
                counter = 0
                for idx in idx_to_code_map:
                    print idx_to_code_map[idx] , "(" + code_to_translation(idx_to_code_map[idx]) + ")"
                    counter += 1
                    if counter > 50:
                        break
            if analysis_type == 'search':
                word = codes[0]
                print word
                for key in code_to_idx_map:
                    if word in code_to_translation(key):
                        print key + " : " + code_to_translation(key)
            if analysis_type == 'neighbors':
                code = codes[0]
                print "\nAnalysis of neighbors for " + code,
                print " (" + code_to_translation(code) + ")",
                print " : Word position at " + str(code_to_idx_map[code])
                print "Top 50" + " cosine distance codes: "
                print "-------------------------------------------------------"
                pool = Pool(threads)
                idx = code_to_idx_map[code]
                distances = pool.map(compute_cosine_distance,[[i,idx] for i in xrange(0,len(lines)-2)])
                idx_to_close_points = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][-50:]
                idx_to_close_points.reverse()
                for top_idx in idx_to_close_points:
                    code = idx_to_code_map[top_idx]
                    print code + " (" + code_to_translation(code) + ") : " + str(distances[top_idx])
            elif analysis_type == 'combine':
                code1 = codes[0]
                code2 = codes[1]
                print "\nAnalysis of combination of concepts : " + code1 + " (" + code_to_translation(code1) + ") and " + code2,
                print "(" + code_to_translation(code2) + ")"
                print "Top 50" + " cosine distance codes: "
                print "-------------------------------------------------------"
                pool = Pool(threads)
                vector = embedding_matrix[code_to_idx_map[code1],:] + embedding_matrix[code_to_idx_map[code2],:]
                distances = pool.map(compute_cosine_distance_with_vector,[[i,vector] for i in xrange(0,len(lines)-2)])
                idx_to_close_points = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][-50:]
                idx_to_close_points.reverse()
                for top_idx in idx_to_close_points:
                    code = idx_to_code_map[top_idx]
                    print code + " (" + code_to_translation(code) + ") : " + str(distances[top_idx])
            elif analysis_type == 'analogy':
                code1 = codes[0]
                code2 = codes[1]
                code3 = codes[2]
                print "\nAnalysis of analogy of concepts ... \n" + code1,
                print " (" + code_to_translation(code1) + ") : " + code2,
                print "(" + code_to_translation(code2) + ") = " + code3 + " (" + code_to_translation(code3) + ") : ?"
                print "Top 50" + " cosine distance codes: "
                print "-------------------------------------------------------"
                pool = Pool(threads)
                vector = embedding_matrix[code_to_idx_map[code2],:] - embedding_matrix[code_to_idx_map[code1],:]
                vector = vector + embedding_matrix[code_to_idx_map[code3],:]
                distances = pool.map(compute_cosine_distance_with_vector,[[i,vector] for i in xrange(0,len(lines)-2)])
                idx_to_close_points = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][-50:]
                idx_to_close_points.reverse()
                print idx_to_close_points
                for top_idx in idx_to_close_points:
                    code = idx_to_code_map[top_idx]
                    print code + " (" + code_to_translation(code) + ") : " + str(distances[top_idx])
        except:
            print "IMPROPER INPUT!!!"
