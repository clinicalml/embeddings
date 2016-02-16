import argparse
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import cosine
from multiprocessing import Pool

#parser = argparse.ArgumentParser()
#parser.add_argument("-ie","-ie")
#ie = parser.parse_args().ie

ie = '../stanford_cuis_svd_300.txt'
threads = 1

#Read in the concept_to_CUI map
concept_to_CUI_hdr = '2b_concept_ID_to_CUI.txt'
concept_to_CUI_map = {}
f = open(concept_to_CUI_hdr, 'r')
lines = f.readlines()
for line in lines:
    concept = line.split('\t')[0]
    CUI = line.split('\t')[1].split('\r')[0]
    concept_to_CUI_map[concept] = CUI

#Read in the CUI_to_type_map
#CUI_to_type_hdr = 'MRSTY.RRF'
#CUI_to_type_map = {}
#f = open(CUI_to_type_hdr, 'r')
#lines = f.readlines()
#for line in lines:
#    CUI = line.split('|')[0]
#    type = line.split('|')[3]
#    if CUI not in CUI_to_type_map:
#        CUI_to_type_map[CUI] = [type]
#    else:
#        CUI_to_type_map[CUI].append(type)

#Read in the concept_to_string map
concept_to_string_hdr = '2a_concept_ID_to_string.txt'
concept_to_string_map = {}
f = open(concept_to_string_hdr, 'r')
lines = f.readlines()
for line in lines:
    concept = line.split('\t')[0]
    string = line.split('\t')[1].split('\r')[0]
    concept_to_string_map[concept] = string

#Create embedding matrix and index_to_concept concept_to_index maps
f = open(ie,'r')
lines = f.readlines()
embedding_matrix = np.zeros(shape=(len(lines)-2,len(lines[2].split(" "))-2))
print embedding_matrix.shape
idx_to_string_map = {}
string_to_idx_map = {}
for i in xrange(2,len(lines)):
    line = lines[i]
    idx = i - 2
    key = line.split(" ")[0]
    embedding = np.array(line.split(" ")[1:-1])
    embedding_matrix[idx,:] = embedding
    idx_to_string_map[idx] = key
    string_to_idx_map[key] = idx


def print_cui_type(CUI):
    return 0

def compute_cosine_distance(idx_pair):
    first_idx = idx_pair[0]
    second_idx = idx_pair[1]
    if sum(abs(embedding_matrix[second_idx,:])) <= 0.00000000001:
        return 0
    if sum(abs(embedding_matrix[first_idx,:])) <= 0.00000000001:
        return 0
    return 1-cosine(embedding_matrix[second_idx,:],embedding_matrix[first_idx,:])

def compute_cosine_distance_with_vector(idx_vector_pair):
    first_idx = idx_vector_pair[0]
    vector = idx_vector_pair[1]
    if sum(abs(vector)) <= 0.00000000001:
        return 0
    if sum(abs(embedding_matrix[first_idx,:])) <= 0.00000000001:
        return 0
    return 1 - cosine(vector,embedding_matrix[first_idx,:])

if __name__ == "__main__":
    while True:
        try:
            inp = str(raw_input('Enter analysis type & the codes (exit to break): '))
            analysis_type = inp.split(" ")[0]
            concepts = inp.split(" ")[1:]
        except:
            inp = ''
        if 'exit' in inp:
            break
        try:
            if analysis_type == 'examples':
                counter = 0
                for idx in idx_to_string_map:
                    print idx_to_string_map[idx] , "(" + concept_to_string_map[idx_to_string_map[idx]] + ")"
                    counter += 1
                    if counter > 50:
                        break
            if analysis_type == 'search':
                inp = concepts[0]
                for key in string_to_idx_map:
                    if inp in concept_to_string_map[key]:
                        print key + " : " + concept_to_string_map[key]
            if analysis_type == 'neighbors':
                inp = concepts[0]
                print "\nAnalysis of neighbors for " + inp,
                concept_string = concept_to_string_map[inp]
                print " (" + concept_string + ")",
                print " : Word position at " + str(string_to_idx_map[inp])
                print "Top 50" + " cosine distance codes: "
                print "-------------------------------------------------------"
                pool = Pool(threads)
                idx = string_to_idx_map[inp]
                distances = pool.map(compute_cosine_distance,[[i,idx] for i in xrange(0,len(lines)-2)])
                idx_to_close_points = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][-50:]
                idx_to_close_points.reverse()
                for top_idx in idx_to_close_points:
                    concept = idx_to_string_map[top_idx]
                    print concept + " (" + concept_to_string_map[concept] + ", " + concept_to_CUI_map[concept] + "), ",
                    #print str(CUI_to_type_map[concept_to_CUI_map[concept]]) + ") : " + str(distances[top_idx])
                    print str(distances[top_idx])
            elif analysis_type == 'combine':
                inp1 = concepts[0]
                inp2 = concepts[1]
                print "\nAnalysis of combination of concepts : " + inp1 + " (" + concept_to_string_map[inp1] + ") and " + inp2,
                print "(" + concept_to_string_map[inp2] + ")"
                print "Top 50" + " cosine distance codes: "
		print "-------------------------------------------------------"
                pool = Pool(threads)
                vector = (embedding_matrix[string_to_idx_map[inp1],:] + embedding_matrix[string_to_idx_map[inp2],:])/2
                distances = pool.map(compute_cosine_distance_with_vector,[[i,vector] for i in xrange(0,len(lines)-2)])
                idx_to_close_points = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][-50:]
                idx_to_close_points.reverse()
                for top_idx in idx_to_close_points:
                    concept = idx_to_string_map[top_idx]
                    print concept + " (" + concept_to_string_map[concept] + ") : " + str(distances[top_idx])
            elif analysis_type == 'analogy':
                inp1 = concepts[0]
                inp2 = concepts[1]
                inp3 = concepts[2]
                print "\nAnalysis of analogy of concepts ... \n" + inp1,
                print " (" + concept_to_string_map[inp1] + ") : " + inp2,
                print "(" + concept_to_string_map[inp2] + ") = " +inp3 + " (" + concept_to_string_map[inp3] + ") : ?"
                print "Top 50" + " cosine distance codes: "
                print "-------------------------------------------------------"
                pool = Pool(threads)
                vector = embedding_matrix[string_to_idx_map[inp2],:] - embedding_matrix[string_to_idx_map[inp1],:]
                vector = vector + embedding_matrix[string_to_idx_map[inp3],:]
                distances = pool.map(compute_cosine_distance_with_vector,[[i,vector] for i in xrange(0,len(lines)-2)])
                idx_to_close_points = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])][-50:]
                idx_to_close_points.reverse()
                for top_idx in idx_to_close_points:
                    concept = idx_to_string_map[top_idx]
                    print concept + " (" + concept_to_string_map[concept] + ") : " + str(distances[top_idx])
        except:
            print "Improper input."
