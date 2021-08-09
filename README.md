# Knowledge-Graph
Thesis on Knowledge Graph Creation, Analytics about tuples, Score of the Graph, Efficiency of prediction
Extension of Knowledge Graph from New Information Source *
Bhargava Rama Raju Dandu
School of Computing
Dublin City University
Dublin, Ireland
Bhargava.Dandu2@mail.dcu.ie
Yashaswi Verma
School of Computing
Dublin City University
Dublin, Ireland
yashaswi.verma2@mail.dcu.ie
Abstract—Knowledge Graph has heat in current research
work after Google released google knowledge Graph in 2012,
which helped in the Google Search engine. After this, The
area of research in the Knowledge graph has drastically
increased. Most researchers have continued their research in
entity linking, Knowledge graph completion, and AI-related
approach. We proposed a novel approach to increment the
knowledge graph dynamically. Lots of information in the world
is unstructured and has to be updated. Knowledge graphs
take advantage in this and can use in question-answering,
natural language understanding, and recommended systems.In
this research paper we incremented the knowledge graph
dynamically.We have evaluated the incremental Knowledge
Graph on the basis of the embedded vectors of the entities.
Keywords- Knowledge Graph (KG), Knowledge Graph
Embeddings (KGE), RDF triples, Subject, Relation, Predicate,
Ontologies, Knowledge Base (KB), MR Rank (Mean
Reciprocal Rank), Margin Ranking Loss (MRL).
1. Introduction
Knowledge graphs are gaining huge popularity due
to their effectiveness in accepting wide range of applications via entity-relationship synopsis. Large,Broad-coverage
knowledge graphs are built by giving human input and
structured data sets. The further process is extracting information from text and then modify the extracted information
using the data analysis algorithms and machine learning
algorithms. Because of rapidly increasing amount of text
now the usage and importance of knowledge graph has
continued to increase rapidly and knowledge graphs can
be used in many applications. Many intelligent applications
use Knowledge Graphs like DBpedia [1] and Free Base [2]
for semantic search and question-answering which gained
huge prosperity and ability of understanding and reasoning
equivalent to human. One feature of Knowledge Graph
that did not receive research level attention is dynamically
increment the Knowledge Graph. In this research paper, we
focus on this topic and considering ways of extending the
Knowledge graph using different methodologies. Speaking
in terms of data emergencies, data related to COVID-19
is perfect example of rapid emergency of the data. So, we
gathered data from the internet and used in our embedding
models to demonstrate the rapid emergency of the content.
We have also built a KG out of that dataset and tried to
increment it dynamically.
Strictly speaking, the definition of a knowledge graph
(KG) is a graph-structured Knowledge Base that stores
knowledge in the form of the relation between entities.These entities and relations are represented in the form
of triples. Generally a Triple has to be represented in a (subject,relation,predicate) structure. In which, (S,R,P) belongs
to Triple T.
In this research paper,we created a Knowledge Base by
extracting triples from the data related to COVID-19 which
is taken from Wikipedia. A custom Knowledge graph has
been created from that Knowledge base using Ampligraph.
[15] Using this knowledge graph any person can get related
information about COVID-19 which is done by using entity
linking of the knowledge graph. Extending the knowledge
graph by updating it regularly is done by KG embeddings so
that, people get updated information regarding this pandemic
disease. For our experiment we extracted triples from the
wikipedia dataset based on 4 keywords, COVID, Pandemic,
Vaccine, Corona. The triples were extracted out of that
dataset using SpaCy and Neural-coref which contained of
5 columns (subject,relation,object,subject-type,object-type).
Many approaches have been proposed in order to create a Knowledge graph from an unstructured data or extracted triples. But we proposed Ampligraph [15] for construction of the custom Knowledge graph.A 2-dimentional
Knowledge graph has been built using NetworkX library
in python.These knowledge graphs can be filtered to see
particular nodes and relations of the triples.
This research paper is organised as follows:First, Introduction to the research paper, Secondly related survey of the
art of constructing a knowledge garph, Thirdly information
about dataset and Knowledge Graph Embedding models,
followed by A complete overview on the Ampligraph,
followed by Evaluations of the Embedding modles and
experiments and methodologies present in the knowledge
graph, results obtained from the experiments, conclusion,
and finally references taken from research papers.
2. Related Surveys
In context of this research topic, we have taken the
related surveys from the following papers.
James P. McCusker et al., has focused on their paper
about the research going on with a knowledge graph with
a basic understanding of the Knowledge graph and their
different use in the field of research [13]. We have learned
the basic terminologies and got a brief idea about the
methods of the knowledge graph. They evaluated a wide
variety of knowledge resources, graphs, and ontologies.
Shaoxiong Ji et al.., have considered in the paper the
essential research topics explaining knowledge graph representation, knowledge acquisition and completion, temporal
knowledge graph, knowledge-aware applications [21]. The
concept of triples and triples extraction has been explained
in detailed in this research paper, from which we have
extracted the entities and relations from the dataset. The
primary study from this tells us about how Knowledge
acquisition aims to construct knowledge graphs from unstructured text, complete an existing knowledge graph, and
discover and recognize entities and relations.
Luca Costabello et al.., build a Package in Python
named AmpliGraph, which deals with Knowledge Acquisition Methods [15]. AmpliGraph is a suite of neural machine
learning models for relational Learning, a branch of machine
learning that deals with supervised Learning on knowledge
graphs. targeted learning, an analytic framework that yields
a double robust effect. The detailed overview of Ampligraph
is given in section 4.
Natthawut Kertkeidkachorn et al.., has focused on their
paper about the research of creating a Knowledge graph
from an unstructured data [10]. The dataset which we obtained is also in unstuctured form, Creating and completion
of the knowledge graph is completely referred from this
research paper. They have evaluated the whole process with
the help of the diagram which is shown in figure 1.
Figure 1. Architecture of Knowledge Graph
Abhilash. T. Bedadur et al..,have constructed a Knowledge graph by transforming the unstructured text into structured text which facilitates the integration and retrieval of
the information [22]. This paper gives complete information
about how to create proper triples by using several techniques like, sentence segmentation and Syntax tree parsing.
By using these techniques we have extracted triples using
NLP.
Pedro Tabacof et al.., has overlooked the problem of
probability calibration and used platt scaling and isotonic
regression which leads to calibrated methods [19]. We get
significantly better results than the uncalibrated models from
all calibration methods.
Alexis Pister and Ghislain Atemezing have been implementing a knowledge graph fact checking using graph
embedding models [17]. They have trained 6 embedding
models DistMult, HolE [5], TransE [16], TransR, ComplEx and RDF2VEC. In which their results indicate that
RDF2VEC gives the higher AUC score of 0.877 for the
prediction of the correctness of the statements. By taking
this research paper as reference, we have implemented the
KGE models in Ampligraph.
Tianxing Wu et and al.., has described their strategy
of integrating the knowledge embeddings of relations and
entities using vector representations with respective to their
context [20]. They also defined their own scoring function
and loss function of the translating model. From this research paper we have taken the methods of how to train
different datasets using evaluation metrics.We got to know
about the Anatomy of the model is main concept for entity
linking.
Ida Szubert and Mark Steedman has focused on modifying the AskNET algorithm, generally the algorithm uses
string similarity tool for matching a node [18]. They have
proposed a modified algorithm which uses vector embeddings instead. From this research paper we came to learn
about different embeddings. The similar embedding we have
used in our paper is graph embeddings and word-based
embeddings for the entity linking of the triples.
Apart from these studies we proposed a novel approach
on how to Dynamically increment the Knowledge graph
using the embedding vectors of the incremented Knowledge
Base consecutively.
3. Datasets and Knowledge Graph Embedding
models
3.1. Datasets(Construction of knowledge base)
Currently, There are many Knowledge Bases such as
WN18RR, FB15k [9] present, which is used as a standard for creating Knowledge Graph and their evaluation
scores to compare them for the graph’s efficiency. These
datasets are open source data available and used for various
researches, including link prediction, semantic search and
entity recognition. Custom Knowledge graphs are gaining
more attention since a large knowledge graph has millions
of entities and relations, making it less efficient. In this
research, we build our custom knowledge base incorporating
facts about corona, COVID, and vaccine, which we scrape
Subject Relation object SubjectType
ObjectType
China reported 140 cases GPE Cardinal
Italy overtook China GPE GPE
OPEC requested Mexico ORG GPE
TABLE 1. EXAMPLES OF TRIPLES USED TO REPRESENT ENTITY TYPES
from Wikipedia pages. We used the Wikipedia API to extract unstructured information from the web pages related
to the topic corona, Vaccine, Pandemic and COVID. The
unstructured information we get from Wikipedia is then
processed using Natural Language Processing Techniques,
including Spcay and NeuralCoref libraries to extract the
subject, predicate, and object (triples)from the unstructured
text.
Neuralcoref is a module that is a pipeline for coreference
resolution to find all entities linked to the text’s same context. Triples are the data structure widely used as a format
to create a knowledge base; these triples are a collection of
entities and their relations extracted from unstructured text.
In the figure 2, we can take an example from the dataset
Like, (corona,symptom,cough) is a Triple within our dataset.
In this example subject is corona, relation is symptom and
object is cough.
Figure 2. An example of triple taken from the dataset.
In the table 1, three triples are taken from the dataset as
an example. The table 1 also illustrates about the subjecttype and also object-type other than subject, relation and
object. For eg., the subject-type of Chine is GPE which
means it is a Geographical Point. Similarly, with Italy and
Mexico also.The object-type cardinal tells that object is
containing numbers. For eg., object 140 cases is cardinal
because it is containing number in it.
3.2. Construction of the Knowledge Graph
The term Knowledge graph is similar to Knowledge base
with a minor difference in which, The knowledge graph
can be viewed as a graph in context of graph structure
whereas, Knowledge base involves the formal semantics and
interpretation of facts.
The custom knowledge base we created now has data
with around 5,164 triples, including 8,130 entities and 2000
relations. These triples are used to create a Knowledge graph
where entities represent nodes in graphs and relationships
as the graph’s edges. The NetworkX package in python
helps visualize the knowledge graph and analyze the graph’s
nodes (entities). It also helps us to filter graphs for particular
entities to look out all the relations to other entities in the
entire knowledge graph [14].
3.3. Knowledge graph embedding models
Knowledge graph embedding models are neural architectures that encodes the concept from knowledge graph
(i.e. entities E and relation types R) into low-dimensional,
continuous vectors Rk. Knowledge graph embeddings are
applied in Knowledge graph completion, entity resolution,
and link-based clustering
3.3.1. TransE. In this research paper, we implemented two
KGE models. TransE is embedding model used to learn the
low-dimensional embeddings of entities and vectors. In every Knowledgebase, hierarchical relationships are common.
Moreover, for representing them, Translating(TransE) model
is used. TransE is the most representational translating
model which represents relations and entities in the form
of vectors in the same space ( P
d
). In TransE model, the
relations are represented as translation in embedding space:
For instance if a triple (h,r,t) holds, then the embedding
of tail entity t, and head entity h, should be close, and in
addition to it, some vectors are dependant on the relationship
r,i.e, h + t = r [3]
if we consider the embedding vectors of head h, tail t,
relation r, then the scoring function is defined as follows:
fh,t = − || h + r − t ||n . (1)
This function takes triple (h,r,t) as an input and returns the
degree of correctness of that triple as an output. Higher
scores are not considered as plausible when compared to
lower scores. For optimizing the embedding vectors of
relations and entities, TransE uses MRL ( Margin Ranking
Loss) function. MRL is used to generate least difference
between positive and negative triple score which is known
as margin. The formula for MRL is given as follows. [16]
L =
XX[Fr(h, t) + Y − Fr(h
0
, t
0
)] (2)
Scoring function of TransE model: The scoring function of TransE relies on distances and computes the similarity between embedding of the subject translated by the
embedding of the predicate and object, using the L1 or L2
norm || . || .
fT ransE = − || es + rp − eo ||n (3)
3.3.2. ComplEx. : The extension of the DistMult model is
ComplEx model. Complex embedding model [6] is used to
handle the symmetric and antisymmetric relation. Complex
model uses only Hermitian dot product which makes it much
simpler than other embedding models like HolE [8]. It is
linear in both space complexity and time complexity. If
the relations given are anti-symmetric then the eigen values
decomposition is not possible , and can be done only in
the complex space. With the complex eigen values the dot
product is also known as hermition product, it is defined as.,
hu, vi := u
−T
v (4)
in equation (4) u and v are vectors.
Scoring funtion of the ComplEx model: The scoring
function of ComplEx uses the extended version of DistMult
by using Hermition dot product.
fComplEx = Re(hrp, es, e¯oi) (5)
4. Ampligraph
4.1. Introduction
There are many tools to built the knowledge graphs in
which Ampligraph is much more flexible with python and
Knowledge graph embedding models.When compared to the
other tools like GRAKN, [11] Ampligraph is much more
flexible for the implementation of the Knowledge Graph
Embeddings (KGE) models. [15] Ampligraph is a suite of
neural machine learning models for relational learning, a
branch of machine learning that generates Knowledge Graph
Embeddings (KGE) and also deals with supervised learning
of knowledge graph. The main uses of Ampligraph are:
• From an existing Knowledge Graph we can create
and discover a new Knowledge graph.
• We can complete the large Knowledge Graphs that
are left Incomplete with missing statements.
• Generate stand-alone Knowledge Graph Embedding
(KGE).
• We can completely create and evaluate a new relational model.
4.2. Anatomy of the model
The neural architecture of a graph is trained to obtain
the Knowledge graph embeddings. In this training phase,
scoring funtion, loss function, must be done to obtain the
best fit model.Ampligraph can have many such components.
They can be used in any combinations to obtain the best
output for the datasets. Ampligraph includes the following
models.
1) Soring function: For each specific model this function assigns a different score for each triple. Scoring
functions of KGE models used in this research
paper:
• ConvKB:The ConvKB [12] funtion is similar to ConvE by containing convolutional
layers, but also uses the dot product.
fConvKB = concat(g([es, rp, eo]) ∗ Ω)).W
(6)
• DistMult: The scoring function of DistMult
uses the tri-linear dot product.
fDistMult = hrp, es, eoi (7)
2) Loss function: The loss functions are used with
KGE models while model selection and are passed
as hyperparameter. The loss functions that are used
in this research paper are:
• PairwiseLoss:
L(θ) = X
t+εG
X
t−ε∂
max(0, [Y + fmodel
(t
−; θ) − fmodel(t
+; θ)]) (8)
In the equation (8) G is set of positives,
∂ is set of corruptions, fmodel(t; θ) is the
model-specific scoring function.
• AbsoluteMarginLoss:
L(θ) = X
t+εG
X
t−ε∂
fmodel
(t
−; θ) − max(0, [Y − fmodel(t
+; θ)]) (9)
In the equation (9) G is set of positives, ∂ is
set of corruptions, fmodel(t; θ) is the modelspecific scoring function. [7]
The init method is common in the above loss
funtions in which number of negatives, margin to
be used in pairwise loss computation are given as
parameters.
5. Evaluation
This module includes metrics to evaluate a neural graph
embedding models, along with negatives generation, [4]
and Applying Learning to rank based evaluation protocols
to KGs which are used in literatures. Learning to rank metrics to evaluate the performance of neural graph embedding
models.
• Rank score: The rank of the triple is rank of positive element against list of negative elements. rank
(s,p,o) = ampligraph.evaluation.rank score(y true,
y pred, pos lab=1).
rank(s,p,o)
(10)
1) Parameters:
y true (ndarray, shape [n])- An array of
binary labels. The array only contains one
positive.
y pred (ndarray, shape [n]) – An array of
scores, for the positive element and the n-1
negatives.
pos lab (int) – The value of the positive label
(default = 1).
2) Returns: It returns the rank of the positives
against the negatives.
3) Return type: int.
• MRR score: The Mean Reciprocal Rank function
is used to compute the Mean of the Reciprocal of
the elements of the vector rankings. This function
is used in addition to the rank evolution protocol
of Ampligraph.evaluation.evaluat performance(). It
is formally defined as follows:
MR =
1
| Q |
X
|Q|
i=1
1
ranks,p,o
(11)
In the equation (11) Q is a set of triples and (s,p,o)
is a triple belongs to Q.
1) Parameters: ranks (ndarray or list, shape [n]
or [n,2]) – Input ranks of n test statements.
2) returns: mrr score
3) return type: float.
• MR score: Mean Rank is a function that computes
the mean of a vector of a ranking. It is defined as
follows:
MR =
1
| Q |
X
|Q|
i=1
ranks,p,o (12)
In equation (12) Q is a set of triples and (s,p,o) is a
triple which belongs to Q.
1) Parameters: ranks (ndarray or list, shape [n]
or [n,2]) – Input ranks of n test statements.
2) returns: mr score
3) return-type:float.
• hits @ n score: Hits@N is a function that
computes how many elements of vector rankings
make it to top N positions. It is formally defined as
follows:
Hits@N =
X
|Q|
i = 11ifranks,p,o ≤ N (13)
In the equation (13) Q is a set of triples and (s,p,o)
is a triple which belongs to Q.
1) Parameters
ranks (ndarray or list, shape [n] or [n,2]) –
Input ranks of n test statements
n (int) – The maximum rank considered to
accept a positive.
2) returns: hits@n score.
3) return-type: float.
Negatives Generation: Negative genaration routines are
corruption strategies based on the Local Closed-World Assumption (LCWA).The following funtions genarates negatives:
generate corruptions for eval(X, . . . [, . . . ]) : Generates
corruption for evaluation.
generate corruptions for fit(X[, . . . ]) : Generates corruption for training.
6. Experiments
At the beginning of our experiment, we compared some
of the trending Knowledge Graph embedding Model, including the one that follows the Convolutional Neural Network
and Compares them by standard evaluation metrics for
custom build knowledge graphs. Which includes MR, MRR,
Hits@10, Hits@3, Hits@1.The embedding Model which
we used are: TransE, DistMult, ComplEX, ConvE, HolE,
CovKB.These Models are trained on similar Hyperparameters where applicable with k=100, Corruption Strategy =
’s+o’,loss=’multiclass nll’,batches count=100.
After getting the result from the above experiment, the
immediate step is to choose the model that gave the best
MR, MRR, and Hits@n value to apply that for the next
examination which is to check the feasibility of adding new
information to the knowledge graph without rebuilding the
whole knowledge graph again.
We first divided our data set into train and test set as
our dataset is having 1564 triple. We don’t need a valid
set to carry out this task. For an experimental setup, we
first take 1000 triple in the train set and 25 triples for the
test set because the embedding model we are using takes
at max 25 test set triples for training 1000 triples. We
got the output that our first experiment has 1626 unique
entities and around 400 unique relations as there are some
entities(subject, object) and relation(predicate) duplicates,
which is acceptable in every scenario. We train those triple
and get the embedding vectors by defining the hyperparameters of the embedding model with k=10 dimensions, optimizer=’adam’,regulizer=’LP,’ loss=’ multiclass nll’ which
gave the best MR(Mean Rank), MRR(Mean Reciprocal
Rank) for the dataset. We store those embedding vectors
in a file for evaluation afterward.
The second step to this experiment is to add 100 triple
extra in the training set, keeping the test set the same. This
adds around 150 entities and 80 relations total in the training
set. We apply the same KGE model on this training set to
see what are the changes in the scores and the embedding
vectors of the previous 1000 triples we kept the same. These
steps are repeated until 1300 triples, adding 100 extra every
time; the embedded vectors and the predicted ranks are
recorded for each experiment. We predict the values for the
test triples using our evaluation matrices.
7. Results
Evaluation of our experiment followed the standard
method: for each triple (h,r,t) in the test set, the head(h) is
replaced by head’(h)’, and we compute the score of (h,r,t’)
and then we rank all of these corrupted triples by the scores.
Multiple corrupted versions of a triple may exist in a dataset
because an entity could have various entity types
KGEM MRR MR Hits@10 Hits@3 Hits@3
TransE 0.04 2151.78 0.06 0.04 0.02
DistMult 0.01 3315.79 0.02 0.01 0.01
ComplEx 0.01 3485.3 0.03 0.01 0.01
HolE 0.001 5277.4 0.02 0.02 0.01
ConvE 0.03 2307.56 0.03 0.03 0,01
ConvKB 0.02 2249.03 0.02 0.02 0.01
TABLE 2. EVALUATION SCORES FOR DIFFERENT KGE MODELS.
In table 2, we compared different KGE models on our
own custom built knowledge graph. By looking at the results
the TransE performs the best with highest MRR 0.04 as this
represents our model is good for the outliers.
ENTITY 1000-1100 1000-1200 1100-1200
Alcohol 0.44304 0.37608 0.38635
Anosmia 0.590685 0.433424 0.56736
Microsoft 0.391619 0.46257 0.397833
Corona Virus 0.5174898 0.520569 0.46053
Illness 0.44121 0.349922 0.26903
TABLE 3. EUCLIDEAN DISTANCE BETWEEN ENTITY VECTORS OF
CONSECUTIVELY INCREMENTED KNOWLEDGE GRAPH
In table 3, we represent the values of the Euclidean
distance between the embedded vectors of the entity which
are present in all the 3 knowledge graphs of 1000, 1100,
1200 triples. This shows the embedding changes as we add
new information to the knowledge graph.
Head Relation Tail
1000 Predictions
SRANK ORANK
1100 Prediction
SRANK ORANK
symptoms include shortness1540 1450 1781 1670
India has deaths 1565 1540 1019 979
Air pollutionincreasesrisk 1232 1419 947 931
symptoms include diarrhea 1368 1223 1749 1565
TABLE 4. PREDICTED RANK OF TEST (1000,1200) TRIPLES
In table 4 and table 5 we checked the test triples ranks
for all the 4 incremented Knowledge graphs by corrupting
subject and object separately represented consecutively by
SRANK and ORANK.
8. Conclusion and Future Work
In this paper , we witnessed the different model accuracy
on the custom knowledge graph. We noticed TransE scores
Head Relation Tail
1200 Predictions
SRANK ORANK
1300 Prediction
SRANK ORANK
symptoms include shortness1727 1510 791 885
India has deaths 913 908 1929 1764
Air pollutionincreasesrisk 1542 1661 1160 1198
symptoms include diarrhea 25 164 1764
TABLE 5. PREDICTED RANK OF TEST (1200,1300) TRIPLES
are better for outliers that are present in our dataset. We
proposed extending the knowledge graph by sets of an
experiment that followed the embedding vector replacement
for the old triples and only kept the new embedded vectors
for new entities introduced to the KG. We compared the
results which show the varied euclidean distance between
the early vectors and the new vector. The predicted rank for
different sizes of KG shows some similarities, which can
be useful for the extension of KG. More research can be
done in the field for Extending KG dynamically by Transfer
Learning as it allows transfer learning without rebuilding the
model. Our method dynamically cannot change the vectors.
Another way to proceed with this approach is to change the
model that will only train on the new triples provided based
on neural networks.
9. Acknowledgement
We acknowledge the guidance and suggestions of the
Professor Gareth Jones
References
[1] S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R.
Cyganiak, and Z. Ives, “Dbpedia: A nucleus for a
web of open data,” in The semantic web, Springer,
2007, pp. 722–735.
[2] K. Bollacker, C. Evans, P. Paritosh, T. Sturge, and
J. Taylor, “Freebase: A collaboratively created graph
database for structuring human knowledge,” in Proceedings of the 2008 ACM SIGMOD international
conference on Management of data, 2008, pp. 1247–
1250.
[3] A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston,
and O. Yakhnenko, “Translating embeddings for modeling multi-relational data,” in Advances in neural information processing systems, 2013, pp. 2787–2795.
[4] M. Nickel, K. Murphy, V. Tresp, and E. Gabrilovich,
“A review of relational machine learning for knowledge graphs,” Proceedings of the IEEE, vol. 104,
no. 1, pp. 11–33, 2015.
[5] M. Nickel, L. Rosasco, and T. Poggio, “Holographic
embeddings of knowledge graphs,” arXiv preprint
arXiv:1510.04935, 2015.
[6] T. Trouillon, J. Welbl, S. Riedel, E. Gaussier, and ´
G. Bouchard, “Complex embeddings for simple link
prediction,” in International Conference on Machine
Learning (ICML), vol. 48, 2016, pp. 2071–2080.
[7] T. Hamaguchi, H. Oiwa, M. Shimbo, and Y. Matsumoto, “Knowledge transfer for out-of-knowledgebase entities: A graph neural network approach,”
arXiv preprint arXiv:1706.05674, 2017.
[8] K. Hayashi and M. Shimbo, “On the equivalence of
holographic and complex embeddings for link prediction,” arXiv preprint arXiv:1702.05563, 2017.
[9] R. Kadlec, O. Bajgar, and J. Kleindienst, “Knowledge base completion: Baselines strike back,” arXiv
preprint arXiv:1705.10744, 2017.
[10] N. Kertkeidkachorn and R. Ichise, “T2kg: An endto-end system for creating knowledge graph from
unstructured text,” in Workshops at the Thirty-First
AAAI Conference on Artificial Intelligence, 2017.
[11] A. Messina, H. Pribadi, J. Stichbury, M. Bucci, S.
Klarman, and A. Urso, “Biograkn: A knowledge
graph-based semantic database for biomedical sciences,” in Conference on Complex, Intelligent, and
Software Intensive Systems, Springer, 2017, pp. 299–
309.
[12] D. Q. Nguyen, T. D. Nguyen, D. Q. Nguyen, and
D. Phung, “A novel embedding model for knowledge
base completion based on convolutional neural network,” arXiv preprint arXiv:1712.02121, 2017.
[13] Q. Wang, Z. Mao, B. Wang, and L. Guo, “Knowledge
graph embedding: A survey of approaches and applications,” IEEE Transactions on Knowledge and Data
Engineering, vol. 29, no. 12, pp. 2724–2743, 2017.
[14] Y. Luan, L. He, M. Ostendorf, and H. Hajishirzi,
“Multi-task identification of entities, relations, and
coreference for scientific knowledge graph construction,” arXiv preprint arXiv:1808.09602, 2018.
[15] L. Costabello, S. Pai, C. L. Van, R. McGrath, N.
McCarthy, and P. Tabacof, AmpliGraph: a Library
for Representation Learning on Knowledge Graphs,
Mar. 2019. DOI: 10.5281/zenodo.2595043. [Online].
Available: https://doi.org/10.5281/zenodo.2595043.
[16] M. Nayyeri, S. Vahdati, J. Lehmann, and H. S. Yazdi,
Soft marginal transe for scholarly knowledge graph
completion, 2019. arXiv: 1904.12211 [cs.AI].
[17] A. Pister and G. A. Atemezing, “Knowledge graph
embedding for triples fact validation.,” in ISWC Satellites, 2019, pp. 21–24.
[18] I. Szubert and M. Steedman, “Node embeddings for
graph merging: Case of knowledge graph construction,” in Proceedings of the Thirteenth Workshop on
Graph-Based Methods for Natural Language Processing (TextGraphs-13), 2019, pp. 172–176.
[19] P. Tabacof and L. Costabello, “Probability calibration for knowledge graph embedding models,” arXiv
preprint arXiv:1912.10000, 2019.
[20] T. Wu, A. Khan, H. Gao, and C. Li, “Efficiently embedding dynamic knowledge graphs,” arXiv preprint
arXiv:1910.06708, 2019.
[21] S. Ji, S. Pan, E. Cambria, P. Marttinen, and P. S. Yu,
A survey on knowledge graphs: Representation, acquisition and applications, 2020. arXiv: 2002.00388
[cs.CL].
[22] A. T. Bedadur and D. S. Vaishali, “Construction of a
knowledge graph for query system,”
