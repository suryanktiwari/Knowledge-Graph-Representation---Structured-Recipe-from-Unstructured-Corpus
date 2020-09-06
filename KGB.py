import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community
import operator
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import sys

recipe = []
name = []
inp = []
other = []
comment = []
qty = []
unit = []

fills = [name, inp, other, comment, qty, unit]
fillnms = ['name', 'inp', 'other', 'comment', 'qty', 'unit']
stop_words=set(stopwords.words("english"))
lem = WordNetLemmatizer()

# This function takes the structured recipe data in JSON and applies pre-processing techniques
# such as lemmataization, lowercasing, removing preceding and trailing spaces on the data.
def preprocess():
    # Pre Processing the json data
    for i in range(df.shape[0]):
        title = df.iloc[i]['title']
        ingredients = df.iloc[i]['ingredients']
        for ing in ingredients:
            recipe.append(title)
            for fill, name in zip(fills, fillnms):
                try:
                    # Converting to lower_case
                    data = ing[name].lower()
                    
                    # Removing Punctuations
                    for c in string.punctuation:
                        data = data.replace(c," ")
                    # Lemmatization
                    temp=''            
                    for word in data.split(' '):
                        if word not in stop_words and len(word)!=1:
                            temp+=lem.lemmatize(word)+' '
                    if temp=='':
                        temp=lem.lemmatize(data)
                    data = temp
                        
                    if data[0]==' ':
                        # Remove preceding spaces
                        j=0
                        while j<len(data):
                            if data[j]==' ':
                                data=data[1:]
                            else:
                                j=len(data)
    
                    if data[-1]==' ':
                        # Remove trailing spaces
                        j=len(data)-1
                        while j>0:
                            if data[j]==' ':
                                data=data[:-1]
                                j-=1
                            else:
                                j=0
                            
                    fill.append(data)
                except:
                    fill.append('') 
                        
    # Pre Processing the recipe list
    i=0
    for name in recipe:
        name = name.lower()
        for c in string.punctuation:
            name = name.replace(c," ")
        name = word_tokenize(name)
        temp=[]
        for word in name:
            if word not in stop_words:
                temp.append(lem.lemmatize(word))
        name = temp
        
        res=''
        for term in name:
            res+=' '+term
            
        # Remove preceding spaces
        j=0
        while j<len(res):
            if res[j]==' ':
                res=res[1:]
            else:
                j=len(res)
        recipe[i]=res
        i+=1

# This function makes an inverted list from every keyword and maps the keywords to a list
# of nodes and edges the keywords can point to.
def make_inverted_index():
    inverted_index = dict()
    excluded = ['inp', 'qty', 'unit']
    # Creating Inverted Index
    for fill, name in zip(fills, fillnms):
        if name not in excluded:
            for entry in fill:
                data = entry.split()
                for term in data:
                    if term in inverted_index:
                        inverted_index[term].add(entry)
                    else:
                        inverted_index[term] = set()
                        inverted_index[term].add(entry)
    
    for r in recipe:
        data = r.split()
        for term in data:
            if term in inverted_index:
                inverted_index[term].add(r)
            else:
                inverted_index[term] = set()
                inverted_index[term].add(r)
    return inverted_index


# Making a dataframe for knowledge graph construction
def make_dataframe():
    d = dict()
    d['recipe']=recipe
    for fill, name in zip(fills, fillnms):
        d[name]=fill
    
    df = pd.DataFrame(data=d)
    #print(df)
    print(df.shape)
    return df

# In this function a Knowledge Graph is created based on the structure obtained from NYTagger.
# Recipe Name is the source vertex, other+name is the target vertex, and comment is the edge
# This triple relation helps in creating the KG
def kg_relations(inverted_index, df):
    source = []
    target = []
    edge = []
    
    for i in range(df.shape[0]):
        src = str(df.iloc[i]['recipe'])
        source.append(src)
        edg = str(df.iloc[i]['comment'])
        edge.append(edg)
        oth = str(df.iloc[i]['other'])
        nam = str(df.iloc[i]['name'])
        tar = oth
        if tar!='' and nam!='':
            tar+=' '+nam
            terms = oth.split()
            for term in terms:
                inverted_index[term].add(tar)
            terms = tar.split()
            for term in terms:
                inverted_index[term].add(tar)
        else:
            tar+=nam
        target.append(tar)
    return source, target, edge, inverted_index

# Creating KG using triples (Source, Target, Edge)
def make_kg(source, target, edge):
    d = {"source":source, "target":target, "edge":edge}
    kg_df = pd.DataFrame(data=d)
    
    G=nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr="edge", create_using=nx.MultiDiGraph())
    #G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="crumbled"], "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    print(G.nodes)
    
    # Plotting the KG
    plt.figure(figsize=(20,20))
    pos = nx.spring_layout(G, k = 0.5)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=500, edge_cmap=plt.cm.Blues, pos = pos)
    nx.draw_networkx_edge_labels(G, pos=pos)    
    plt.show()
    
    return G, kg_df

# Computation of popular recipes and ingredients
def return_data(centrality, recipe_names):
    popular_recipe = []
    popular_ingredient = []
    for item in centrality.keys():
        if item in recipe_names: 
            popular_recipe.append(item)
        else:
            popular_ingredient.append(item)
            
    return popular_recipe, popular_ingredient

# Computation of centrality measures from the Knowledge Graph
def measure_centrality(G, recipe_names):
    deg_centrality = dict(sorted(nx.degree_centrality(G).items(), key = operator.itemgetter(1), reverse=True)[:20])
    #print("\nDegree Centrality: ",deg_centrality)
    
    popular_recipe, popular_ingredient = return_data(deg_centrality, recipe_names)
            
    print("\nMost extensive recipes are: ",popular_recipe)
    print("\nMost popular ingredients are: ",popular_ingredient)
    
    between_centrality = dict(sorted(nx.betweenness_centrality(G).items(), key = operator.itemgetter(1), reverse=True)[:20])
    #print("\n\nBetween Centrality: ",between_centrality)
    
    popular_recipe, popular_ingredient = return_data(between_centrality, recipe_names)
    
    print("\nMost Important recipes that bind other recipes: ",popular_recipe)
    print("\nMost Cohesive & Important ingredients that serve as articulation points are: ",popular_ingredient)
    
    eigen_centrality = dict(sorted(nx.eigenvector_centrality_numpy(G).items(), key = operator.itemgetter(1), reverse=True)[:15])
    
    popular_recipe, popular_ingredient = return_data(eigen_centrality, recipe_names)
    print("\nRelated Influence: Recipes that are related to popular recipes are:  ",popular_recipe)
    print("\nRelated Influence: Ingridients that are used with the most popular ingredients are: ",popular_ingredient)
    
    #print("\nEigen Vector Centrality: ",eigen_centrality)
    #print("\nNumber of nodes Veg Wrap is connected to: ",nx.degree(G, 'Veg Wrap'))

# This function draws several infereces from the KG that has been constructed
def draw_inferences(G, df):
    recipe_names = df['recipe'].unique()
    print("Recipes :\n",len(recipe_names))
    for i in range(len(recipe_names)):
        print(i, recipe_names[i])
    #print(G.edges)
    print("\nNumber of Nodes in the Graph: ",G.number_of_nodes())
    print("\nNumber of Edges in the Graph: ",G.number_of_edges())
    
    cc = list(nx.connected_components(G))
    print("\nConnected Components in the Graph: ",len(cc))
    #print(cc)
    i=0
    print("Inferences for Each Genre created: ")
    for component in cc:
        print("\nComponent:",i+1)
        i = i+1
        H = G.subgraph(list(component))
        measure_centrality(H, recipe_names)
        print("Recipes in This Genre are: \n")
        for item in list(component):
            if item in recipe_names:
                print(item)
    
    largest = max(nx.connected_components(G), key=len)
    print("\nNumber of nodes in the Largest Connected Component of the Graph: ",len(largest))
    
    print("Inferences for the Entire Graph")
    measure_centrality(G, recipe_names)
    partitions = community.best_partition(G)
    #print("Partitions of the Graph: ",len(list(partitions)))
    
    #Draw Communities here: 
    size = float(len(set(partitions.values())))
    p = nx.spring_layout(G)
    count = 0.
    for subcommunity in set(partitions.values()) :
        count = count + 1.
        node = [nodes for nodes in partitions.keys()
                                    if partitions[nodes] == subcommunity]
        nx.draw_networkx_nodes(G, p, node, node_size = 20,
                                    node_color = str(count / size))


    nx.draw_networkx_edges(G, p, alpha=0.5)
    plt.show()


# Main Function
# Check the number of arguments during the script time
# Argument 1 is the name of the json file produced in previous step
if len(sys.argv) < 2 or len(sys.argv) > 2:
    sys.stderr.write('Usage: KGB.py NAME_OF_JSON_FILE')
    sys.exit(1)

# This takes the input to the JSON file containing the structured Recipe
df = pd.read_json (sys.argv[1])
#df = pd.read_json ('100_recipes')
preprocess()
inverted_index = make_inverted_index()
df = make_dataframe()

source, target, edge, inverted_index = kg_relations(inverted_index, df)
G, kg_df = make_kg(source, target, edge)
Gu = nx.MultiDiGraph.to_undirected(G)
draw_inferences(Gu, df)

# Query System
while True:
    query = input("Enter Query\n")
    if query=="exit":
        break
    
    # Query Preprocessing
    query = query.lower()
    query = query.split(' ')
    
    # Lemmatization
    temp=[]
    for word in name:
        if word not in stop_words:
            temp.append(lem.lemmatize(word))
    name = temp
    
    print(query)
    
    # Creating an empty graph for output
    T=nx.empty_graph(0,create_using=nx.MultiDiGraph())

    for q_term in query:
        if q_term in inverted_index:
            
            # Fetch the nodes and edges corresponding to each term
            candidates = inverted_index[q_term]
            print('Candidates:', candidates)
            for term in candidates:
                #  Load the edges with term name, if they exist
                try:
                    H=nx.from_pandas_edgelist(kg_df[kg_df['edge']==term], "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())        
                    T = nx.compose(H,T)
                except:
                    pass
                
                # Converting to an undirected graph
                Gu = nx.MultiDiGraph.to_undirected(G)
        
                #  Try to load the node and it's neighbours if they exist
                neighs=[]        
                try:
                    neighs = list(Gu.neighbors(term))
                except:
                    pass
                for nb in neighs:
                    H = G.subgraph([term, nb])
                    T = nx.compose(H,T)
                    
    # If the graph is not empty then plot the graph
    if not nx.is_empty(T):
        plt.figure(figsize=(15,15))
        pos = nx.spring_layout(T, k = 2)
        nx.draw(T, with_labels=True, node_color='skyblue', node_size=500, edge_cmap=plt.cm.Blues, pos = pos)
        edge_labels=nx.draw_networkx_edge_labels(T,pos=pos)
        plt.show()
    else:
        print('Query keywords not found')
    


