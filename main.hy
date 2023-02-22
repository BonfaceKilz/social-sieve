;; main.hy --- 

;; Copyright (C) 2023  <me@bonfacemunyoki.com>

;; Author: Munyoki Kilyungi <me@bonfacemunyoki.com>

;; This program is free software; you can redistribute it and/or
;; modify it under the terms of the GNU General Public License
;; as published by the Free Software Foundation; either version 3
;; of the License, or (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program. If not, see <http://www.gnu.org/licenses/>.

(import
  nltk
  nltk.corpus [stopwords]
  nltk.tokenize [RegexpTokenizer]
  numpy :as np
  sklearn.neighbors [NearestNeighbors]
  sklearn.feature_extraction.text [CountVectorizer TfidfTransformer]
  os
  re
  spacy
  scipy [spatial]
  scipy.sparse [coo_matrix])

(require
  hyrule.argmove [->>])


(setv *nlp* (spacy.load "en_core_web_lg"))

(setv *stop-words* (sorted (set (stopwords.words "english"))))

(setv *label-names* ["genetic" "science" "covid" "politics" "business"
                     "sports" "genome" "research" "cooking" "gossip"
                     "entertainment" "lifestyle" "travel" "tourism"])

(defmacro nltk-download [name]
  "Check if CORPUS exists.  If it doesn't download it."
  `(let [home-dir (get os.environ "HOME")
         tokenizer-path (os.path.join home-dir
                                      "nltk_data/tokenizers"
                                      (+ ~name ".zip"))
         corpus-path (os.path.join home-dir
                                   "nltk_data/corpora"
                                   (+ ~name ".zip"))]
     (when (not (or (os.path.exists tokenizer-path)
                    (os.path.exists corpus-path)))
       (nltk.download ~name))))

(defn clean-data [document]
  "Remove stop words from DOCUMENT"
  (let [document (.join
                   " "
                   (lfor word (.split
                                (->> document
                                     (re.sub "[^a-zA-Z]" " ")
                                     (.lower)
                                     (re.sub "&lt;/?.*?&gt;" " &lt;&gt; "))
                                " ")
                         :if (not (in word *stop-words*))
                         word))]
    (.join " " (lfor token (*nlp* document)
                     token.lemma_))))
 
(defn extract-top-n-feature-names [document top-n]
  "Use TF-IDF (term frequency/inverse document frequency) to extract
  keywords from a DOCUMENT.  TF-IDF lists word frequency scores
  that highlight words that are more important to the context rather
  than those that appear frequently across documents."

  (defn sort-coo [coo-matrix]
    (let [zipped-data (zip coo-matrix.col
                           coo-matrix.data)]
      (sorted zipped-data
              :key (fn [x] #((get x 1) (get x 0)))
              :reverse True)))
  
  (defn extract-top-n-from-vector
    [feature-names sorted-items top-n]
    (let [items (cut sorted-items 0 top-n)
          score-vals []
          feature-vals []
          results {}]
      (for [[idx score] items]
        (list.append score-vals (round score 3))
        (list.append feature-vals (get feature-names idx)))
      (for [idx (range (len feature-vals))]
        (setv (get results (get feature-vals idx))
              (get score-vals idx))) 
      results))
  
  (let [count-vect (CountVectorizer)
        X (.fit_transform count_vect [document])
        tfidf-transformer (TfidfTransformer :smooth_idf True :use_idf True)
        feature-names (.get_feature_names_out count-vect)
        tfidf-vector (do (.fit tfidf-transformer X)
                         (.transform tfidf-transformer
                                     (.transform
                                       count-vect [document])))]
    (extract-top-n-from-vector
      feature-names
      (sort-coo (.tocoo tfidf-vector))
      top-n)))

(defn embed [tokens model]
  "Return the centroid of the embeddings that correspond to the
  provided TOKENS give some language MODEL.  Any tokens that are
  out-of-vocabulary will not be considered, and stopwords will also be
  excluded. If none of the tokens are valid, the function will return
  an array of zeros." 
  (let [lexemes (lfor token
                      tokens
                      (get model.vocab token))
        vectors (np.asarray
                  (lfor lexeme lexemes
                        :if (and
                              lexeme.has_vector
                              (not lexeme.is_stop)
                              (> (len lexeme.text) 1))
                        lexeme.vector))]
    (if (> (len vectors) 0)
        (.mean vectors :axis 0)
        (np.zeros (get (get model.meta "vectors") "width")))))

(defn centroid->label [centroid labels model]
  "Return the closest label listed in LABELS from the CENTROID and
  MODEL"
  (let [label-vectors (np.asarray
                        (lfor label
                              labels
                              (embed (.split label) model)))
        closest-neighbor (NearestNeighbors
                           :n_neighbors 1
                           :metric spatial.distance.cosine)
        closest-label ((do (.fit closest-neighbor label-vectors)
                           closest-neighbor.kneighbors)
                        [centroid]
                        :return_distance False)]
    
    (get labels (get closest-label 0 0))))


(defn main []
  (nltk-download "stopwords")
  (nltk-download "wordnet")
  (nltk-download "punkt")
  (let [centroid (embed (.keys (extract-top-n-feature-names
                                 (clean-data "Hermes beats forecasts on robust growth in China, U.S. http://reut.rs/3k1Eb6q")
                                 10))
                        *nlp*)]
    (print
      (centroid->label centroid *label-names* *nlp*))))

;; Run the program to classify things
(main)
