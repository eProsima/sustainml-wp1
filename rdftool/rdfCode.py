from rdflib import Graph, Namespace, RDF, Literal
from rdflib.namespace import XSD

def get_mlgoals(graph_path):
    ###########################################################
    ### get types of machine learning goals:                ###
    ###########################################################
    import rdflib
    g = rdflib.Graph()
    g.parse(graph_path)
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?problem
    WHERE {
      ?problem a conn:Problem .
    }
    """
    results = g.query(query)
    problems = [row[0] for row in results]
    return problems

def get_cover_tags(graph_path):
    ###########################################################
    ### get cover tags (modalities) of machine learning:    ###
    ###########################################################
    import rdflib
    g = rdflib.Graph()
    g.parse(graph_path)
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?coverTag
    WHERE {
      ?coverTag a conn:CoverTag .
    }
    """
    results = g.query(query)
    cover_tags = [row[0] for row in results]
    return cover_tags

def get_models(mlgoal, graph_path):
    ###########################################################
    ### get models with correct machine learning goal:      ###
    ###########################################################
    from rdflib import Graph

    # Load your RDF graph
    g = Graph()
    g.parse(graph_path, format="turtle")

    # Query to find all models
    problem_literal = Literal(mlgoal, datatype=XSD.string)

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX model: <http://example.org/model/>
    SELECT ?model ?downloads
    WHERE {
      ?model a conn:Model .
      ?model conn:hasProblem ?problem .
      ?model conn:downloads ?downloads .
      FILTER (?problem = ?problem_literal)
    }
    ORDER BY DESC(?downloads)
    """

    results = g.query(query, initBindings={'problem_literal': problem_literal})
    models = [(str(row[0]), int(row[1])) for row in results]
    return models


def get_model_info(suggested_models, graph_path):
    ###########################################################
    ### get info about model                                ###
    ###########################################################
    import rdflib
    g = rdflib.Graph()
    g.parse(graph_path)

    model_infos = []

    for model_name, _ in suggested_models:
        model_literal = Literal(model_name, datatype=XSD.string)

        query = """
        PREFIX conn: <http://example.org/conn/>
        PREFIX model: <http://example.org/model/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?model ?id ?name ?problem ?coverTag ?library ?downloads ?likes ?lastModified
        WHERE {
          ?model a conn:Model .
          ?model conn:model_name ?name .
          ?model conn:model_id ?id .
          ?model conn:hasProblem ?problem .
          ?model conn:hasCoverTag ?coverTag .
          ?model conn:usesLibrary ?library .
          ?model conn:downloads ?downloads .
          ?model conn:likes ?likes .
          ?model conn:lastModified ?lastModified .
          FILTER (?name = ?model_literal)
        }
        """

        results = g.query(query, initBindings={'model_literal': model_literal})
        for row in results:
            details = {
                'model_uri': row[0],
                'id': row[1],
                'name': row[2],
                'problem': row[3],
                'coverTag': row[4],
                'library': row[5],
                'downloads': row[6],
                'likes': row[7],
                'lastModified': row[8]
            }
            model_infos.append(details)

    return model_infos
