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
        "OPTIONS": {
            # Wait for a competing writer instead of failing immediately with
            # "database is locked" (background run threads write concurrently).
            "timeout": 30,
            # Take the write lock at BEGIN so read->write upgrades inside
            # atomic() blocks cannot deadlock (Django >= 5.1).
            "transaction_mode": "IMMEDIATE",
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
        },
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
TARGET_LLM_PROVIDER = os.environ.get("TARGET_LLM_PROVIDER", "openai")
# Only OpenAI-style providers are supported; an old .env value (e.g. "custom")
# must not become the default for new evaluations.
if TARGET_LLM_PROVIDER not in ("openai", "openai_compatible", "evaair", "virgin"):
    TARGET_LLM_PROVIDER = "openai"
EXTERNAL_RESPONSE_API_KEY = os.environ.get("EXTERNAL_RESPONSE_API_KEY", "")
try:
    TARGET_LLM_TIMEOUT_SECONDS = float(os.environ.get("TARGET_LLM_TIMEOUT_SECONDS", "120"))
except (TypeError, ValueError):
    TARGET_LLM_TIMEOUT_SECONDS = 120.0
try:
    # Retries after the first attempt (timeouts, 429, 5xx, connection errors).
    TARGET_LLM_MAX_RETRIES = max(int(os.environ.get("TARGET_LLM_MAX_RETRIES", "2")), 0)
except (TypeError, ValueError):
    TARGET_LLM_MAX_RETRIES = 2
try:
    # Parallel target LLM calls per run; 1 keeps calls strictly sequential.
    TARGET_LLM_CONCURRENCY = max(int(os.environ.get("TARGET_LLM_CONCURRENCY", "1")), 1)
except (TypeError, ValueError):
    TARGET_LLM_CONCURRENCY = 1
try:
    # A run fails as a whole when the first N generations it attempts all fail
    # (e.g. wrong base URL / model). Later isolated failures are only recorded.
    TARGET_LLM_FAILFAST_GENERATIONS = max(
        int(os.environ.get("TARGET_LLM_FAILFAST_GENERATIONS", "5")), 1
    )
except (TypeError, ValueError):
    TARGET_LLM_FAILFAST_GENERATIONS = 5
try:
    # External-bot runs have no worker thread; mark them failed only after
    # this many hours without any POSTed responses.
    EXTERNAL_RUN_STALE_HOURS = float(os.environ.get("EXTERNAL_RUN_STALE_HOURS", "24"))
except (TypeError, ValueError):
    EXTERNAL_RUN_STALE_HOURS = 24.0

# The prompt agent reuses the target LLM network limits unless overridden.
try:
    PROMPT_AGENT_TIMEOUT_SECONDS = float(
        os.environ.get("PROMPT_AGENT_TIMEOUT_SECONDS", str(TARGET_LLM_TIMEOUT_SECONDS))
    )
except (TypeError, ValueError):
    PROMPT_AGENT_TIMEOUT_SECONDS = TARGET_LLM_TIMEOUT_SECONDS
try:
    PROMPT_AGENT_MAX_RETRIES = max(
        int(os.environ.get("PROMPT_AGENT_MAX_RETRIES", str(TARGET_LLM_MAX_RETRIES))), 0
    )
except (TypeError, ValueError):
    PROMPT_AGENT_MAX_RETRIES = TARGET_LLM_MAX_RETRIES

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

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    "loggers": {
        "apps": {
            "handlers": ["console"],
            "level": os.environ.get("APP_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
        "services": {
            "handlers": ["console"],
            "level": os.environ.get("APP_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}
