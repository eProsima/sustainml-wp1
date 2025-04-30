from rdflib import Graph, Namespace, RDF, Literal
from rdflib.namespace import XSD

def load_graph(file_path):
    ###########################################################
    ### load and parse the graph file:                      ###
    ###########################################################
    g = Graph()
    g.parse(file_path, format="turtle")
    return g

def get_cover_tags(graph):
    ###########################################################
    ### get cover tags (modalities) of machine learning:    ###
    ###########################################################
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?coverTag
    WHERE {
      ?coverTag a conn:CoverTag .
    }
    """
    results = graph.query(query)
    cover_tags = [row[0] for row in results]
    return cover_tags

def get_problems(graph):
    ###########################################################
    ### get types of machine learning problem:              ###
    ###########################################################
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?problem
    WHERE {
      ?problem a conn:Problem .
    }
    """
    results = graph.query(query)
    problems = [row[0] for row in results]
    return problems

def get_problems_for_cover_tag(graph, cover_tag):
    ###########################################################
    ### get problem type from modality:                     ###
    ###########################################################
    CONN = Namespace("http://example.org/conn/")
    cover_tag_literal = Literal(cover_tag, datatype=XSD.string)

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX problem: <http://example.org/problem/>
    SELECT ?problem
    WHERE {
      ?problem a conn:Problem .
      ?problem conn:hasCoverTag ?coverTag .
      FILTER (?coverTag = ?cover_tag)
    }
    """

    results = graph.query(query, initBindings={'cover_tag': cover_tag_literal})
    problems = [row[0] for row in results]
    return problems

def get_modalities_input(graph):
    ###########################################################
    ### get modalities inputs machine learning:             ###
    ###########################################################
    query = """
    PREFIX modality: <http://example.org/modality/>
    PREFIX conn: <http://example.org/conn/>

    SELECT DISTINCT ?modality
    WHERE {
        ?type modality:hasInput ?modality ;
    }
    """
    results = graph.query(query)
    modalities_input = [row[0] for row in results]
    return modalities_input

def get_modalities_output(graph):
    ###########################################################
    ### get modalities outputs machine learning:            ###
    ###########################################################
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT DISTINCT ?modality
    WHERE {
        ?type modality:hasOutput ?modality .
    }
    """
    results = graph.query(query)
    modalities_output = [row[0] for row in results]
    return modalities_output

def get_all_metrics(graph):
    ###########################################################
    ### get all types of metrics:                           ###
    ###########################################################
    query = """
    PREFIX conn: <http://example.org/conn/>  # Add conn prefix
    PREFIX metric: <http://example.org/metric/>
    SELECT DISTINCT ?metric
    WHERE {
        ?metric a metric:Metric.
    }
    """
    results = graph.query(query)
    metrics = {str(metric[0]) for metric in results}
    return metrics


def find_metrics_by_model(graph, model_name):
    ###########################################################
    ### get metrics for a model:                            ###
    ###########################################################
    query = f"""
    PREFIX metric: <http://example.org/metric/>
    PREFIX conn: <http://example.org/conn/>

    SELECT DISTINCT ?metric
    WHERE {{
        ?model a conn:Model ;
               conn:model_name "{model_name}"^^xsd:string ;
               metric:hasMetric ?metric .
    }}
    """
    results = graph.query(query)


    metrics = [str(row[0]) for row in results]
    return metrics

def search_metrics_by_input_modalities(graph, input_modality):
    ###########################################################
    ### get metrics for a input modality:                  ###
    ###########################################################
    problems = find_problem_by_input_modality(graph, input_modality)
    metrics_for_all_problems = {}

    for problem in problems:
        models = get_models_for_problem(graph, problem)
        models_with_metrics = {}

        for model,downloads in models:
            metrics = find_metrics_by_model(graph, model)
            models_with_metrics[model] = metrics

        metrics_for_all_problems[problem] = models_with_metrics

    return metrics_for_all_problems

def search_metrics_by_modalities(graph, input_modality, output_modality):
    ###########################################################
    ### get metrics for a modality:                         ###
    ###########################################################
    problems = find_problem_by_modalities(graph, input_modality, output_modality)
    metrics_for_all_problems = {}

    for problem in problems:
        models = get_models_for_problem(graph, problem)
        models_with_metrics = {}

        for model,downloads in models:
            metrics = find_metrics_by_model(graph, model)
            models_with_metrics[model] = metrics

        metrics_for_all_problems[problem] = models_with_metrics

    return metrics_for_all_problems

def get_models_with_higher_score(graph, metric_name, dataset, score_threshold):
    ###########################################################
    ### get models with the higher scores:                  ###
    ###########################################################
    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX metric: <http://example.org/metric/>
    SELECT ?model
    WHERE {
        ?model a conn:Model .
        ?metric a metric:Metric .
        ?model metric:hasMetric ?metric .
        ?metric metric:metricName ?metricName .
        ?metric metric:onDataset ?dataset .
        ?metric metric:hasScore ?score .
        FILTER (xsd:float(?score) > ?score_threshold)
    }
    """

    # convert score to literal
    score_threshold_literal = Literal(score_threshold, datatype=XSD.float)

    results = graph.query(
        query,
        initBindings={
            'metricName': Literal(metric_name),
            'dataset': Literal(dataset),
            'score_threshold': score_threshold_literal
        }
    )

    models = [str(row[0]) for row in results]
    return models

def find_problem_by_modalities(graph, input_modality, output_modality):
    ###########################################################
    ### get problem from modalities:                        ###
    ###########################################################
    query = f"""
    PREFIX modality: <http://example.org/modality/>
    PREFIX conn: <http://example.org/conn/>

    SELECT DISTINCT ?problem
    WHERE {{
        ?problem a conn:Problem ;
                 modality:hasInput "{input_modality}"^^xsd:string ;
                 modality:hasOutput "{output_modality}"^^xsd:string .
    }}
    """
    results = graph.query(query)

    problems = [str(row[0]) for row in results]
    return problems

def find_problem_by_input_modality(graph, input_modality):
    ###########################################################
    ### get problem from input_modality:                    ###
    ###########################################################
    query = f"""
    PREFIX modality: <http://example.org/modality/>
    PREFIX conn: <http://example.org/conn/>

    SELECT DISTINCT ?problem
    WHERE {{
        ?problem a conn:Problem ;
               modality:hasInput "{input_modality}"^^xsd:string .
    }}
    """
    results = graph.query(query)

    problems = [str(row[0]) for row in results]
    return problems

def get_models_with_max_size(graph, max_parameters=None):
    ###########################################################
    ### get models threshold by the size:                   ###
    ###########################################################
    query = """
    PREFIX conn: <http://example.org/conn/>
    SELECT ?model
    WHERE {
        ?model a conn:Model .
        ?model conn:parameters ?parameters .
        FILTER (xsd:integer(?parameters) >= ?min_parameters)
        """

    if max_parameters is not None:
        query += "FILTER (xsd:integer(?parameters) <= ?max_parameters)"

    query += "}"  # Close the WHERE clause

    max_parameters_literal = Literal(max_parameters, datatype=XSD.integer)

    results = graph.query(query, initBindings={'max_parameters': max_parameters_literal})
    models = [str(row[0]) for row in results]
    return models

def get_models_for_problem(graph, problem_literal_text):
    ###########################################################
    ### get models with correct machine learning goal:      ###
    ###########################################################
    problem_literal = Literal(problem_literal_text, datatype=XSD.string)

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

    results = graph.query(query, initBindings={'problem_literal': problem_literal})
    models = [(row[0], row[1]) for row in results]
    return models

def get_models_for_problem_and_tag(graph, problem_literal_text, tag):
    ###########################################################
    ### get models with correct machine learning goal and   ###
    ### with the specified tag (e.g., transformers)         ###
    ###########################################################
    problem_literal = Literal(problem_literal_text, datatype=XSD.string)
    tag_literal = Literal(tag, datatype=XSD.string)

    query = """
    PREFIX conn: <http://example.org/conn/>
    PREFIX model: <http://example.org/model/>
    SELECT ?model ?downloads
    WHERE {
      ?model a conn:Model .
      ?model conn:hasProblem ?problem .
      ?model conn:hasTag ?modelTag .
      ?model conn:downloads ?downloads .
      FILTER (?problem = ?problem_literal && ?modelTag = ?tag_literal)
    }
    ORDER BY DESC(?downloads)
    """

    results = graph.query(query, initBindings={'problem_literal': problem_literal, 'tag_literal': tag_literal})
    models = [(row[0], row[1]) for row in results]
    return models

def get_model_details(graph, model_name):
    ###########################################################
    ### get info about model                                ###
    ###########################################################
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

    results = graph.query(query, initBindings={'model_literal': model_literal})
    details = {}
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
    return details

def print_results(literals, label):
    ###########################################################
    ### print on terminal <label> information.              ###
    ###########################################################
    print(f"List of available {label}:")
    for literal in literals:
        print(literal)
    print()


def print_models(models):
    ###########################################################
    ### print on terminal types of model.                   ###
    ###########################################################
    print("Models ordered by downloads:")
    for model, downloads in models:
        print(f"Model: {model}, Downloads: {downloads}")
    print()


def print_model_details(details):
    ###########################################################
    ### print on terminal details of a model.               ###
    ###########################################################
    print("Model details:")
    for key, value in details.items():
        print(f"{key}: {value}")
    print()

def print_metrics_for_problem(metrics_for_all_problems):
    ###########################################################
    ### print metrics of a problem:                         ###
    ###########################################################
    for problem, metrics_list in metrics_for_all_problems.items():
        print(f"Problem: {problem}")
        unique_metrics = set()
        for model, metrics in metrics_list.items():
            for metric in metrics:
                unique_metrics.add(metric)

        for metric in unique_metrics:
            print(f"  Metric: {metric}")
    print()
