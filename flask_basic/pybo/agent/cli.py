# pybo/agent/cli.py
from __future__ import annotations
import click
from flask import current_app
from pybo.rag.engine import rag_sync_env_pdf_dir

def register_cli(app):
    @app.cli.command("rag.sync")
    def _rag_sync():
        "RAG_PDF_DIR의 새/변경 PDF를 동기화합니다."
        with app.app_context():
            res = rag_sync_env_pdf_dir()
            click.echo(res)
