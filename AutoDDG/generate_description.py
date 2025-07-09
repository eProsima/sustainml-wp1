# Adapted from:
# Zhang, H., Liu, Y., Hung, W.-L., Santos, A., & Freire, J. (2025).
# "AutoDDG: Automated Dataset Description Generation using Large Language Models".
# arXiv:2502.01050. https://doi.org/10.48550/arXiv.2502.01050

import json
import re

class DatasetDescriptionGenerator:
    def __init__(self, client, model_name, temperature=0.0, description_words=100):
        """
        Initializes the DatasetDescriptionGenerator with the OpenAI client and model parameters.

        :param client: OpenAI client for making requests.
        :param model: The model to use for generating the description (default: gpt-3.5-turbo-0125).
        :param temperature: Temperature for controlling randomness in the generation (default: 0.3).
        :param description_words: Target number of words for the description (default: 100).
        """
        self.client = client  # Use the client instance
        self.model = model_name
        self.temperature = temperature
        self.description_words = description_words
        print(f"Dataset Description Generator initialized with model: {model_name}, temperature: {temperature}, description words: {description_words}")
    
    def _fix_json_response(self, response_text):
            """
            Automatically close all open braces by counting mismatched '{' and '}' in the response text.

            :param response_text: The response text to fix.
            :return: The fixed response text.
            """
            response_text = re.search(r'\{.*\}', response_text, re.DOTALL).group()
            
            # Append the required number of closing braces
            open_braces = response_text.count('{')
            close_braces = response_text.count('}')
            response_text += '}' * (open_braces - close_braces)

            # Use regex to remove any trailing comma before the final closing brace
            response_text = re.sub(r',\s*}', '}', response_text)
            return response_text
    
    def _generate_prompt(self, dataset_sample, 
                         dataset_profile=None, use_profile=False,
                         semantic_profile=None, use_semantic_profile=False,
                         data_topic=None, use_topic=False):
        """
        Generates the prompt for the OpenAI model based on the provided inputs.

        :param dataset_sample: Sample of the dataset to include in the description.
        :param dataset_profile: Optional dataset profile to include in the description.
        :param use_profile: Boolean flag to include the dataset profile.
        :param semantic_profile: Optional semantic types to include in the description.
        :param use_semantic_profile: Boolean flag to include semantic types.
        :param instruction: Additional instruction for generating the description.
        :return: The generated prompt as a string.
        """

        template = """
        {
        "description": "A brief description of the dataset, including its purpose, content, and any relevant context.",
        "profile": "A detailed profile of the dataset in natural sentences, including its structure, data types, and any relevant metadata.",
        "topic": "A concise topic that best describes the dataset's primary theme, ideally in 2-3 words.",
        "keywords": "A list of relevant keywords that can be used for indexing and search purposes.",
        "applications": "A description of potential applications or use cases for the dataset, highlighting its relevance to specific domains or industries, in list format."
        }
        """
        prompt = f"Answer the question using the following information.\n"

        # Add dataset sample
        prompt += f"First, consider the dataset sample:\n\n{dataset_sample}\n"

        # Add dataset profile
        if use_profile and dataset_profile:
            prompt += (
                f"Additionally, the dataset profile is as follows:\n\n{dataset_profile}\n\n"
                f"Based on this profile, please add sentence(s) to enrich the dataset description.\n\n"
            )

        # Add semantic types if the flag is set to True
        if use_semantic_profile and semantic_profile:
            prompt += (
                f"Furthermore, the semantic profile of the dataset columns is as follows:\n{semantic_profile}\n\n"
                "Based on this information, please add sentence(s) discussing the semantic profile in the description.\n\n"
            )

        # Add data topic if the flag is set to True
        if use_topic and data_topic:
            prompt += (
                f"Moreover, the dataset topic is: {data_topic}. "
                f"Based on this topic, please add sentence(s) describing what this dataset can be used for.\n\n"
            )

        prompt += (f"If each row contains a large number of numerical values (e.g., 784 or 3072) within the range [0, 255], it may represent pixel values."
                   f"If these values can be reshaped into a 2D or 3D array (e.g., 28×28 or 32×32×3), the data likely represents images."
                   f"If it represents images, guess which well-known dataset it could be (e.g., MNIST, CIFAR-10, Fashion-MNIST, etc.), based on size, shape, number of values, label structure, and pixel ranges.")
        prompt += (
            f"Question: Based on the information above and the requirements, provide a dataset metadata in json format."
            f"Use only metadata information, formatting in json. Only return the json, nothing else."
            f"For the json use the following template {template}\n\n"
        )

        
        return prompt
    
    def generate_description(self, dataset_sample, 
                             dataset_profile=None, use_profile=False,
                             semantic_profile=None, use_semantic_profile=False,
                             data_topic=None, use_topic=False):
        """
        Generates a dataset description using the provided dataset sample, profile, and semantic types.

        :param dataset_sample: Sample of the dataset to include in the description.
        :param dataset_profile: Optional dataset profile.
        :param use_profile: Boolean flag to include the dataset profile in the description.
        :param semantic_profile: Optional semantic types for dataset columns.
        :param use_semantic_profile: Boolean flag to include semantic types in the description.
        :param instruction: Additional instruction for the description generation.
        :return: Generated description as a string.
        """
        # Create the prompt using the provided parameters
        prompt = self._generate_prompt(dataset_sample, dataset_profile, use_profile,
                                       semantic_profile, use_semantic_profile,
                                       data_topic, use_topic)
        
        # Make a request to the OpenAI API to generate the description
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an assistant for a dataset search engine. Your goal is to improve the readability of dataset description for dataset search engine users."},
                {"role": "user", "content": prompt}
            ],
            options={
            "temperature": 0.0
             }
            #temperature=self.temperature
        )
        
        # Extract the response content
        description = response['message']['content']
        print(description)
        description_json = self._fix_json_response(description)
        return prompt, description_json

class SemanticProfiler:
    TEMPLATE = """
    {'Temporal': 
        {
            'isTemporal': Does this column contain temporal information? Yes or No. NOTE: If the column comes from an image (e.g., pixel values, color channels, etc.), the answer must be 'No', even if the data appears sequential',
            'resolution': If Yes, specify the resolution (Year, Month, Day, Hour, etc.).
        },
     'Spatial': {'isSpatial': Does this column contain spatial information? Yes or No,
                 'resolution': If Yes, specify the resolution (Country, State, City, Coordinates, etc.).},
     'Entity Type': What kind of entity does the column describe? (e.g., Person, Location, Organization, Product),
     'Domain-Specific Types': What domain is this column from (e.g., Financial, Healthcare, E-commerce, Climate, Demographic),
     'Function/Usage Context': How might the data be used (e.g., Aggregation Key, Ranking/Scoring, Interaction Data, Measurement).}
    """

    RESPONSE_EXAMPLE = """
    {
    "Domain-Specific Types": "General",
    "Entity Type": "Temporal Entity",
    "Function/Usage Context": "Aggregation Key",
    "Spatial": {"isSpatial": false,
                "resolution": ""},
    "Temporal": {"isTemporal": true,
                "resolution": "Year"}
    }
    """

    def __init__(self, client, model_name="llama3"):
        self.client = client  # Use the client instance
        self.model = model_name
        print(f"Semantic Type Analyzer initialized with model: {model_name}")

    def _fix_json_response(self, response_text):
            """
            Automatically close all open braces by counting mismatched '{' and '}' in the response text.

            :param response_text: The response text to fix.
            :return: The fixed response text.
            """
            response_text = re.search(r'\{.*\}', response_text, re.DOTALL).group()
            
            # Append the required number of closing braces
            open_braces = response_text.count('{')
            close_braces = response_text.count('}')
            response_text += '}' * (open_braces - close_braces)

            # Use regex to remove any trailing comma before the final closing brace
            response_text = re.sub(r',\s*}', '}', response_text)
            return response_text
    
    def get_semantic_type(self, column_name, sample_values):
        prompt = f"""
        You are a dataset semantic analyzer. Based on the column name and sample values, classify the column into multiple semantic types. 
        Please group the semantic types under the following categories: 
        'Temporal', 'Spatial', 'Entity Type', 'Data Format', 'Domain-Specific Types', 'Function/Usage Context'. 
        Following is the template {self.TEMPLATE}
        Please follow these rules:
        1. The output must be a valid JSON object that can be directly loaded by json.loads. Example response is {self.RESPONSE_EXAMPLE}
        2. All keys from the template must be present in the response.
        3. All keys and string values must be enclosed in double quotes.
        4. There must be no trailing commas.
        5. Use booleans (true/false) and numbers without quotes.
        6. Do not include any additional information or context in the response.
        7. If you are unsure about a specific category, you can leave it as an empty string.

        Column name: {column_name}
        Sample values: {sample_values}
        """
        
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in dataset semantic analysis."},
                {"role": "user", "content": prompt}
            ],
            options={
            "temperature": 0.0
             }
        )
        
        response_text = response['message']['content']

        response_text = self._fix_json_response(response_text)

        try:
            semantic_dict = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Failed to parse GPT response as JSON for column: {column_name}")
            print(f"Response text: {response_text}")
            semantic_dict = None
        print(f"Semantic analysis for column '{column_name}': {semantic_dict}")
        return semantic_dict

    def analyze_dataframe(self, dataframe):
        """
        Analyzes a pandas DataFrame and returns semantic types for each column.

        :param dataframe: pandas DataFrame to be analyzed.
        :return: Dictionary of semantic types for each column.
        """
        def _get_sample(data_pd, sample_size):
            if sample_size < len(data_pd):
                data_sample = data_pd.sample(sample_size, random_state=9)
            else:
                data_sample = data_pd
            return data_sample

        semantic_summary = []
        dataframe_sample = _get_sample(dataframe, 5)

        # Iterate through the columns to profile each one
        for column in dataframe.columns:
            try:
                # Get the first few values as a sample to provide context
                sample_values = dataframe_sample[column].astype(str).tolist()

                # Call GPT to get the semantic type
                semantic_description = None
                retry_count = 0
                while semantic_description is None and retry_count < 3:
                    if retry_count > 0:
                        print(f"Retrying for column: {column}")
                    semantic_description = self.get_semantic_type(column, sample_values)
                    retry_count += 1
                if retry_count == 3:
                    print(f"Failed to get semantic type for column: {column}")
                    continue
                # print(column, semantic_description)

                # Create a human-readable summary for the column
                column_summary = f"**{column}**: "
                entity_type = semantic_description.get('Entity Type', 'Unknown')
                if entity_type != '' and entity_type != 'Unknown':
                    column_summary += f"Represents {entity_type.lower()}. "

                # Handle spatial and temporal cases
                isTemporal = semantic_description['Temporal'].get('isTemporal', False)            
                if isTemporal and semantic_description['Temporal']['isTemporal'] == True:
                    column_summary += f"Contains temporal data (resolution: {semantic_description['Temporal']['resolution']}). "
                isSpatial = semantic_description['Spatial'].get('isSpatial', False)
                if isSpatial and semantic_description['Spatial']['isSpatial'] == True:
                    column_summary += f"Contains spatial data (resolution: {semantic_description['Spatial']['resolution']}). "
                
                domain_type = semantic_description.get('Domain-Specific Types', 'Unknown')
                if domain_type != '' and domain_type != 'Unknown':
                    column_summary += f"Domain-specific type: {domain_type.lower()}. "

                function_context = semantic_description.get('Function/Usage Context', 'Unknown')
                if function_context != '' and function_context != 'Unknown':
                    column_summary += f"Function/Usage context: {function_context.lower()}. "

                # Add sample values
                # if sample_values:
                #     sample_str = ', '.join(sample_values)
                #     column_summary += f"Sample values include {sample_str}."

                # Append the summary for this column
                semantic_summary.append(column_summary)
            except:
                continue

        # Join the semantic summary into a readable format
        final_summary = "The key semantic information for this dataset includes:\n" + '\n'.join(semantic_summary)
        return final_summary