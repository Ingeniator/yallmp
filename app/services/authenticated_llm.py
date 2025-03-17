# from app.services.token_manager import OIDCTokenManager
# from app.core.config import settings

# async def get_authenticated_model():
#     if settings.llm_authorization_type == "APIKEY":
#         return FakeOpenAIChatModel()
#     elif settings.llm_authorization_type == "BEARER":
#         token_manager = OIDCTokenManager(settings.llm_oidc_authorization_url, settings.llm_oidc_client_id, settings.llm_oidc_client_secret)
#         token = await token_manager.get_token()
#         # Example LangChain model
#         return FakeOpenAIChatModel(
#             #model="gpt-4",
#             #openai_api_base="https://your-openai-compatible-server.com/v1",
#             #openai_api_key="",  # No API key needed
#             request_headers={"Authorization": f"Bearer {token}"}
#         )
#     elif settings.llm_authorization_type == "CERT":
#         return FakeOpenAIChatModel(
#             #model="gpt-4",
#             #openai_api_base="https://your-openai-compatible-server.com/v1",
#             #openai_api_key="",  # No API key needed
#             #cert_file_path=settings.llm_client_cert_file,
#             #cert_key_file_path=settings.llm_client_key_file
#         )
#     else:
#         raise ValueError(f"Unsupported authorization type: {settings.llm_authorization_type}")
