import logging

import httpx
from databricks.sdk import WorkspaceClient

class DatabricksClient:
    def __init__(self, databricks_host: str, serving_endpoint_name:str, request_timeout: float = 300):
        self.databricks_host = databricks_host
        self.serving_endpoint_name = serving_endpoint_name
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout),
        )

    async def exchange_token(self, provider_oauth_token: str):

        url = f"{self.databricks_host}/oidc/v1/token"

        data = {
            "subject_token": provider_oauth_token,  # replace with your JWT token variable
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "scope": "all-apis"
        }

        response = await self.client.post(url, data=data)

        return response.json()['access_token']

    def create_databricks_payload(self, text:str, history: list, stream: bool = False):

        messages = list(history)

        messages.append({"role": "user", "content": text})

        payload = {"input": messages}

        if stream:
            payload["stream"] = stream

        return payload

    def parse_model_output(self, response):
        parsed_output = []

        logging.info(f"Response from openai client: {response}")
        logging.info(f"Response output from openai client: {response.output}")

        for item in response.output:
            logging.info(f"Item type: {item.type}")
            if item.type == "message":
                parsed_output.append({"type": "message", "text": "".join([e.text for e in item.content])})
            if item.type == "function_call":
                parsed_output.append({"type": "tool_call", "arguments": item.arguments, "name": item.name, "call_id": item.call_id})
            if item.type == "function_call_output":
                parsed_output.append(
                    {"type": "tool_result", "call_id": item.call_id, "output": item.output})

        logging.info(f"Response parsed from openai client: {parsed_output}")
        return parsed_output

    async def call_model_endpoint(self, text:str, provider_oauth_token: str, history:list, stream: bool = False):

        oauth_db_token = await self.exchange_token(provider_oauth_token)

        workspace_client = WorkspaceClient(host=self.databricks_host, token=oauth_db_token)

        openai_client = workspace_client.serving_endpoints.get_open_ai_client()

        logging.info(f"Actual history in the chatbot: {history}")

        payload = self.create_databricks_payload(text, history, stream=stream)

        response = openai_client.responses.create(model=self.serving_endpoint_name, **payload)

        parsed_output = self.parse_model_output(response)

        return parsed_output