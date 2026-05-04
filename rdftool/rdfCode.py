from neo4j import GraphDatabase

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# Connect to Neo4j
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

###############################################################################
### Problem-specific WHITELIST                                              ###
### Only these models will be considered for the listed problems.           ###
### Model names must match the `m.name` in the DB exactly (case-sensitive). ###
###############################################################################
WHITELIST = {
    "summarization": frozenset([
        "ARTeLab/mbart-summarization-mlsum",
        "Ameer05/bart-large-cnn-samsum-rescom-finetuned-resume-summarizer-10-epoch-tweak-lr-8-100-1",
        "BEE-spoke-data/pegasus-x-base-synthsumm_open-16k",
        "DunnBC22/pegasus-multi_news-NewsSummarization_BBC",
        "ELiRF/NASES",
        "EbanLee/kobart-summary-v3",
        "Einmalumdiewelt/PegasusXSUM_GNAD",
        "IlyaGusev/mbart_ru_sum_gazeta",
        "IlyaGusev/rugpt3medium_sum_gazeta",
        "JordiAb/BART_news_summarizer",
        "JustinDu/BARTxiv",
        "KamilAin/bart-base-booksum",
        "KipperDev/bart_summarizer_model",
        "KoddaDuck/autotrain-text-summa-38210101165",
        "Madan490/finetuned_multi_news_bart_text_summarisation",
        "Mahalingam/DistilBart-Med-Summary",
        "Mbilal755/Radiology_Bart",
        "MohamedZaitoon/bart-fine-tune",
        "NTUYG/ComFormer",
        "RUCAIBox/mvp",
        "z-dickson/bart-large-cnn-climate-change-summarization",
    ]),
    "translation": frozenset([
        "AI-Sweden-Models/gpt-sw3-6.7b-v2-translator",
        "Abdulmohsena/Faseeh",
        "Babelscape/mrebel-base",
        "Babelscape/mrebel-large",
        "Babelscape/mrebel-large-32",
        "Biniam/en_ti_translate",
        "BlackKakapo/opus-mt-en-ro",
        "BlackKakapo/opus-mt-ro-en",
        "BubbleSheep/Hgn_trans_en2zh",
        "CLAck/en-km",
        "CLAck/en-vi",
        "CLAck/indo-mixed",
        "DevWorld/Gemago-2b",
        "DunnBC22/opus-mt-zh-en-Chinese_to_English",
        "HPLT/translate-en-ar-v1.0-hplt",
        "HPLT/translate-en-hr-v1.0-hplt_opus",
        "HackerMonica/nllb-200-distilled-600M-en-zh_CN",
        "HelpMumHQ/AI-translator-eng-to-9ja",
        "Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU",
        "Helsinki-NLP/opus-mt-ROMANCE-en",
        "Helsinki-NLP/opus-mt-ar-de",
        "Helsinki-NLP/opus-mt-ar-en",
        "Helsinki-NLP/opus-mt-ar-es",
        "Helsinki-NLP/opus-mt-ar-fr",
        "Helsinki-NLP/opus-mt-ar-tr",
    ]),
    "text-generation": frozenset([
        "0Tick/danbooruTagAutocomplete",
        "0Tick/e621TagAutocomplete",
        "0x7o/BulgakovLM-3B",
        "0x7o/pyGPT-50M",
        "2early4coffee/DialoGPT-medium-deadpool",
        "4eJIoBek/ruGPT3_small_nujdiki_stage1",
        "A2/kogpt2-taf",
        "zen-E/deepspeed-chat-step2-model-opt350m",
        "zen-E/deepspeed-chat-step3-rlhf-actor-model-opt1.3b",
        "zlsl/en_l_warhammer_fantasy",
        "zlsl/en_l_wh40k_full",
        "zlsl/l_erotic_kink_chat",
        "zlsl/l_soft_erotic",
        "zlsl/l_soft_erotic_tm",
        "zlsl/l_warhammer3",
        "zlsl/l_wh40k_all",
        "zlsl/m_cosmos",
        "zlsl/ru_startrek",
        "zlsl/ru_warcraft",
        "zlsl/ru_warhammer40k",
        "zyayoung/cv-full-paper",
    ]),
}

def _apply_whitelist(models, problem_name):
    """
    models: list[ (model_name:str, downloads:Any) ]
    problem_name: the Problem.name string passed by the UI/engine
    returns: filtered list if the problem has a whitelist; otherwise unchanged
    """
    allowed = WHITELIST.get(problem_name)
    if not allowed:
        return models
    # keep original order (already sorted by downloads), just drop non-allowed
    return [(m, d) for (m, d) in models if m in allowed]

def load_graph():
    ###########################################################
    ### make sure neo4j graph is loaded:                    ###
    ###########################################################
    try:
        with neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count LIMIT 1")
            count = result.single()["node_count"]
            # print(f"Neo4j connected successfully. Total nodes: {count}")
        return True
    except Exception as e:
        # print(f"Error connecting to Neo4j: {e}")
        return False

def execute_cypher_query(cypher_query):
    ###########################################################
    ### executes the Cypher query on the Neo4j database:    ###
    ###########################################################
    with neo4j_driver.session() as session:
        results = session.run(cypher_query)
        data = [dict(record) for record in results]
        print(f"Retrieved {len(data)} records from Neo4j")
        return data

def get_cover_tags():
    ###########################################################
    ### get cover tags (modalities) of machine learning:    ###
    ###########################################################
    cypher_query = """
    MATCH (ct:CoverTag)
    RETURN ct.name AS coverTag
    ORDER BY ct.name
    """

    results = execute_cypher_query(cypher_query)
    cover_tags = [record["coverTag"] for record in results]
    return cover_tags

def get_problems():
    ###########################################################
    ### get types of machine learning problem:              ###
    ###########################################################
    cypher_query = """
    MATCH (p:Problem)
    RETURN p.name AS problem
    ORDER BY p.name
    """

    results = execute_cypher_query(cypher_query)
    problems = [record["problem"] for record in results]
    return problems

def get_problems_for_cover_tag(cover_tag):
    ###########################################################
    ### get problem type from modality:                     ###
    ###########################################################
    cypher_query = """
    MATCH (ct:CoverTag {name: $cover_tag})
          <-[:HAS_COVER_TAG]-(m:Model)
          -[:HAS_PROBLEM]->(p:Problem)
    RETURN DISTINCT p.name AS problem
    ORDER BY p.name
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query, cover_tag=cover_tag)
        problems = [record["problem"] for record in results]

    return problems

def get_modalities_input():
    ###########################################################
    ### get modalities inputs machine learning:             ###
    ###########################################################
    cypher_query = """
    MATCH (p:Problem)-[:HAS_INPUT]->(m:Modality)
    RETURN DISTINCT m.name AS modality
    ORDER BY m.name
    """

    results = execute_cypher_query(cypher_query)
    modalities_input = [record["modality"] for record in results]
    return modalities_input

def get_modalities_output():
    ###########################################################
    ### get modalities outputs machine learning:            ###
    ###########################################################
    cypher_query = """
    MATCH (p:Problem)-[:HAS_OUTPUT]->(m:Modality)
    RETURN DISTINCT m.name AS modality
    ORDER BY m.name
    """

    results = execute_cypher_query(cypher_query)
    modalities_output = [record["modality"] for record in results]
    return modalities_output

def get_all_metrics():
    ###########################################################
    ### get all types of metrics:                           ###
    ###########################################################
    cypher_query = """
    MATCH (metric:Metric)
    RETURN DISTINCT metric.name AS metric
    ORDER BY metric.name
    """

    results = execute_cypher_query(cypher_query)
    metrics = {record["metric"] for record in results}
    return metrics


def find_metrics_by_model(model_name):
    ###########################################################
    ### get metrics for a model:                            ###
    ###########################################################
    cypher_query = """
    MATCH (m:Model {name: $model_name})-[:EVALUATED_BY]-(metric:Metric)
    RETURN DISTINCT metric.name AS metric
    ORDER BY metric.name
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query, model_name=model_name)
        metrics = [record["metric"] for record in results]

    return metrics

def search_metrics_by_cover_tag(cover_tag):
    ###########################################################
    ### get metrics for a specific cover tag:               ###
    ###########################################################
    problems = get_problems_for_cover_tag(cover_tag)
    metrics_for_all_problems = {}

    for problem in problems:
        models = get_models_for_problem(problem)
        models_with_metrics = {}

        for model, downloads in models:
            metrics = find_metrics_by_model(model)
            models_with_metrics[model] = metrics

        metrics_for_all_problems[problem] = models_with_metrics

    return metrics_for_all_problems

def search_metrics_by_input_modalities(input_modality):
    ###########################################################
    ### get metrics for a input modality:                  ###
    ###########################################################
    problems = find_problem_by_input_modality(input_modality)
    metrics_for_all_problems = {}

    for problem in problems:
        models = get_models_for_problem(problem)
        models_with_metrics = {}

        for model, downloads in models:
            metrics = find_metrics_by_model(model)
            models_with_metrics[model] = metrics

        metrics_for_all_problems[problem] = models_with_metrics

    return metrics_for_all_problems

def search_metrics_by_modalities(input_modality, output_modality):
    ###########################################################
    ### get metrics for a modality:                         ###
    ###########################################################
    problems = find_problem_by_modalities(input_modality, output_modality)
    metrics_for_all_problems = {}

    for problem in problems:
        models = get_models_for_problem(problem)
        models_with_metrics = {}

        for model, downloads in models:
            metrics = find_metrics_by_model(model)
            models_with_metrics[model] = metrics

        metrics_for_all_problems[problem] = models_with_metrics

    return metrics_for_all_problems

def get_models_with_higher_score(metric_name, dataset, score_threshold):
    ###########################################################
    ### get models with the higher scores:                  ###
    ###########################################################
    cypher_query = """
    MATCH (m:Model)-[:EVALUATED_BY]-(metric:Metric)-[:ON_DATASET]->(d:Dataset {name: $dataset})
    WHERE metric.name = $metric_name
      AND metric.score > $score_threshold
    RETURN DISTINCT m.name AS model
    ORDER BY metric.score DESC
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query,
                            metric_name=metric_name,
                            dataset=dataset,
                            score_threshold=float(score_threshold))
        models = [record["model"] for record in results]

    return models

def find_problem_by_modalities(input_modality, output_modality):
    ###########################################################
    ### get problem from modalities:                        ###
    ###########################################################
    cypher_query = """
    MATCH (p:Problem)-[:HAS_INPUT]->(input:Modality)
    MATCH (p)-[:HAS_OUTPUT]->(output:Modality)
    WHERE input.name = $input_modality AND output.name = $output_modality
    RETURN DISTINCT p.name AS problem
    ORDER BY p.name
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query,
                            input_modality=input_modality,
                            output_modality=output_modality)
        problems = [record["problem"] for record in results]

    return problems

def find_problem_by_input_modality(input_modality):
    ###########################################################
    ### get problem from input_modality:                    ###
    ###########################################################
    cypher_query = """
    MATCH (p:Problem)-[:HAS_INPUT]->(input:Modality)
    WHERE input.name = $input_modality
    RETURN DISTINCT p.name AS problem
    ORDER BY p.name
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query, input_modality=input_modality)
        problems = [record["problem"] for record in results]

    return problems

def get_models_with_max_size(max_parameters=None):
    ###########################################################
    ### get models threshold by the size:                   ###
    ###########################################################
    cypher_query = """
    MATCH (m:Model)
    WHERE m.parameters IS NOT NULL
    """

    if max_parameters is not None:
        cypher_query += " AND m.parameters <= $max_parameters"

    cypher_query += """
    RETURN m.name AS model
    ORDER BY m.parameters DESC
    """

    with neo4j_driver.session() as session:
        if max_parameters is not None:
            results = session.run(cypher_query, max_parameters=max_parameters)
        else:
            results = session.run(cypher_query)
        models = [record["model"] for record in results]

    return models

def get_models_for_problem(problem_literal_text):
    ###########################################################
    ### get models with correct machine learning goal:      ###
    ###########################################################
    cypher_query = """
    MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem)
    WHERE p.name = $problem_name
    RETURN m.name AS model, m.downloads AS downloads
    ORDER BY m.downloads DESC
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query, problem_name=problem_literal_text)
        models = [(record["model"], record["downloads"]) for record in results]

    models = _apply_whitelist(models, problem_literal_text)
    return models

def get_models_for_problem_and_tag(problem_literal_text, tag):
    ###########################################################
    ### get models with correct machine learning goal and   ###
    ### with the specified tag (e.g., transformers)         ###
    ###########################################################
    cypher_query = """
    MATCH (m:Model)-[:HAS_PROBLEM]->(p:Problem)
    MATCH (m)-[:HAS_TAG]->(t:Tag)
    WHERE p.name = $problem_name AND t.name = $tag_name
    RETURN m.name AS model, m.downloads AS downloads
    ORDER BY m.downloads DESC
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query,
                            problem_name=problem_literal_text,
                            tag_name=tag)
        models = [(record["model"], record["downloads"]) for record in results]
    # return models

    # Should be updated in case we come back to filtering by tag, the following line commented out and the previous one uncommented
    return get_models_for_problem(problem_literal_text)

def get_model_details(model_name):
    ###########################################################
    ### get info about model                                ###
    ###########################################################
    cypher_query = """
    MATCH (m:Model)
    WHERE m.name = $model_name
    OPTIONAL MATCH (m)-[:HAS_PROBLEM]->(p:Problem)
    OPTIONAL MATCH (m)-[:HAS_COVER_TAG]->(ct:CoverTag)
    OPTIONAL MATCH (m)-[:USES_LIBRARY]->(l:Library)
    RETURN
      m.name AS name,
      m.id AS id,
      p.name AS problem,
      ct.name AS coverTag,
      l.name AS library,
      m.downloads AS downloads,
      m.likes AS likes,
      m.lastModified AS lastModified
    """

    with neo4j_driver.session() as session:
        results = session.run(cypher_query, model_name=model_name)
        record = results.single()

        if record:
            details = {
                'name': record["name"],
                'id': record["id"],
                'problem': record["problem"],
                'coverTag': record["coverTag"],
                'library': record["library"],
                'downloads': record["downloads"],
                'likes': record["likes"],
                'lastModified': record["lastModified"]
            }
        else:
            details = {}

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
