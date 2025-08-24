FROM python:3.11-slim

# Instala UV
RUN pip install uv

# Instala JupyterLab usando UV (usa requirements.txt si tienes dependencias extra)
RUN uv pip install --system jupyterlab

# Crea carpetas de trabajo
WORKDIR /workspace
RUN mkdir -p /workspace/notebooks /workspace/sharedmodels

# Expone el puerto de JupyterLab
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=${JUPYTER_TOKEN}"]