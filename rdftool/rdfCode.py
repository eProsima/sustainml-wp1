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
            result.single()
        return True
    except Exception:
        return False

def execute_cypher_query(cypher_query):
    ###########################################################
    ### executes the Cypher query on the Neo4j database:    ###
    ###########################################################
    with neo4j_driver.session() as session:
        results = session.run(cypher_query)
        data = [dict(record) for record in results]
        return data


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


