import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_CACHE_DIR = BASE_DIR / ".cache"
PROJECT_MODEL_DIR = BASE_DIR / "models"
for directory in (PROJECT_CACHE_DIR, PROJECT_MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Keep runtime-generated model and NLP data inside this checkout.
os.environ["XDG_CACHE_HOME"] = str(PROJECT_CACHE_DIR)
os.environ["NLTK_DATA"] = str(PROJECT_MODEL_DIR / "nltk")
os.environ["TORCH_HOME"] = str(PROJECT_MODEL_DIR / "torch")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(PROJECT_MODEL_DIR / "sentence-transformers")
os.environ["MPLCONFIGDIR"] = str(PROJECT_CACHE_DIR / "matplotlib")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "insecure-dev-key-change-me")

DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() == "true"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "apps.evaluations",
    "apps.prompt_agents",
    "apps.prompts",
    "apps.llms",
    "apps.runs",
    "apps.responses",
    "apps.fairness",
    "apps.personas",
    "apps.benchmark",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

STATIC_URL = "static/"

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 100,
    # The local frontend is a first-party API client, not a Django form.
    # Disable session auth so POST/PUT/PATCH requests are not rejected by CSRF
    # when a browser happens to carry a Django session cookie.
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
}

CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

# Prompt Generation Agent
PROMPT_AGENT_API_KEY = os.environ.get("PROMPT_AGENT_API_KEY", "")
PROMPT_AGENT_BASE_URL = os.environ.get("PROMPT_AGENT_BASE_URL", "")
PROMPT_AGENT_MODEL_NAME = os.environ.get("PROMPT_AGENT_MODEL_NAME", "")
PROMPT_AGENT_PROVIDER = os.environ.get("PROMPT_AGENT_PROVIDER", "openai_compatible")

# Target LLM under evaluation
TARGET_LLM_API_KEY = os.environ.get("TARGET_LLM_API_KEY", "")
TARGET_LLM_BASE_URL = os.environ.get("TARGET_LLM_BASE_URL", "")
TARGET_LLM_MODEL_NAME = os.environ.get("TARGET_LLM_MODEL_NAME", "")
TARGET_LLM_PROVIDER = os.environ.get("TARGET_LLM_PROVIDER", "openai_compatible")
EXTERNAL_RESPONSE_API_KEY = os.environ.get("EXTERNAL_RESPONSE_API_KEY", "")
try:
    TARGET_LLM_TIMEOUT_SECONDS = float(os.environ.get("TARGET_LLM_TIMEOUT_SECONDS", "420"))
except (TypeError, ValueError):
    TARGET_LLM_TIMEOUT_SECONDS = 120.0

# Default language used by LangFair model selection. The frontend may override
# this per evaluation; supported values are "en" and "auto".
LANGFAIR_DEFAULT_LANGUAGE = os.environ.get("LANGFAIR_DEFAULT_LANGUAGE", "en")

# MatrAIx Persona service
MATRAIX_BASE_URL = os.environ.get("MATRAIX_BASE_URL", "http://127.0.0.1:8765")
MATRAIX_DEFAULT_PERSONA_POOL = os.environ.get(
    "MATRAIX_DEFAULT_PERSONA_POOL",
    "persona/datasets/matraix-persona-1m",
)
try:
    MATRAIX_TIMEOUT_SECONDS = float(os.environ.get("MATRAIX_TIMEOUT_SECONDS", "15"))
except (TypeError, ValueError):
    MATRAIX_TIMEOUT_SECONDS = 15.0
try:
    MATRAIX_CONNECT_TIMEOUT_SECONDS = float(
        os.environ.get("MATRAIX_CONNECT_TIMEOUT_SECONDS", "3")
    )
except (TypeError, ValueError):
    MATRAIX_CONNECT_TIMEOUT_SECONDS = 3.0
