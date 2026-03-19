# use slim python base image
FROM python:3.13-slim

# set the working directory
WORKDIR /app

# copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# install dependencies
RUN pip install uv
RUN uv sync --frozen --no-dev

# copy project files
COPY . .

# expose the port
EXPOSE 8000

# command to run API with uvicorn
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


