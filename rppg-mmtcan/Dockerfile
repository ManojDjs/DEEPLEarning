FROM python:3.12-slim

WORKDIR /code
 
# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6



# Copy the project files for depenedency and env management

COPY pyproject.toml poetry.lock /code/
# Install poetry and project depenedencies with poetry
RUN pip install poetry

RUN poetry install --no-root 

# Copy app folder into the working directory with app folder similar to our project structure
COPY /src /code/src
# COPY /src/models /src/models
# COPY /src/static /src/static
# COPY /src/utils /src/utils
# COPY /src/models/pre_trained_models /src/models/pre_trained_models
# COPY /src/models/pre_trained_models/BP4D_PseudoLabel_DeepPhys.pth /src/models/pre_trained_models/

# expose port to access the api
EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.main:app" ,"--host", "0.0.0.0", "--port", "8000","--workers","4" ]