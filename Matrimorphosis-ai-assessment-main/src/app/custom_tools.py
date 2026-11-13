"""Tasks 7 and 8: four tools. Names, arguments, outputs, errors, metric and log names are FIXED (the autograder checks them).

The heavy parts are PROVIDED (kb/store.py: stats / outline / delete_document, kb/pipeline.py: ingest_file).
You write the tool wrappers: signature + docstring (the docstring is the prompt: say WHEN to use it, describe each argument),
validation, ToolException with `ERROR[CODE]: what happened. Next step: what to do.`, the metric, the log line, `handle_tool_error = True`.
All four are `async def` (blocking calls go through asyncio.to_thread). They act on the CALLING USER's knowledge base:
`runtime.context.user_id` (task 6), never on another user's. build_custom_tools(kb) returns all four.
Look at kb/tools.py for the pattern (ToolRuntime, ToolException, handle_tool_error).

---------------------------------------------------------------------------------- task 7.1  kb_stats(top_keywords: int = 5) -> str
  output  "<D> documents, <S> sections, <C> chunks. Top keywords: k1, k2, ..."   from kb.stats(user_id); at most `top_keywords` keywords
          empty knowledge base: "The knowledge base is empty."
  error   top_keywords outside 1..20  ->  ERROR[INVALID_INPUT]

---------------------------------------------------------------------------------- task 7.2  document_outline(doc_name: str) -> str
  The structure of one document: use it when the user asks what a document contains, how it is organised, or which section covers
  something (then search that section). Resolve doc_name with kb.resolve_doc_name(user_id, doc_name), find the doc_id in
  kb.documents(user_id), read the sections with kb.outline(user_id, doc_id).
  output  first line   "Outline of '<exact file name>' (<N> sections):"          (always the word "sections")
          then one line per section, in reading order, 1-based position:
              "<i>. [L<level>] <heading> (p.<a>-<b>): <summary>"
          the page part is " (p.<a>-<b>)" when page_start != page_end, " (p.<a>)" when they are equal, and left out when the
          document has no pages (page_start is None); then ": <summary>"
  error   no such document in THIS user's knowledge base  ->  ERROR[UNKNOWN_DOCUMENT]  (list the valid names in the message)

---------------------------------------------------------------------------------- task 7.3  observability, every call, ok AND error
  metric  kb_stats: counter "kbstats.calls"    document_outline: counter "outline.calls"
          record with attributes {"status": "ok" | "error"}  (Prometheus: agent_outline_calls_total, ...)
          metrics.get_meter("agent").create_counter(...)   (see how app/telemetry.py creates TOOL_CALLS)
  log     log.info("kb_stats_done", tool="kb_stats", status=..., duration_ms=<int>)
          log.info("document_outline_done", tool="document_outline", status=..., duration_ms=<int>, sections=<int, 0 on error>)
          (`from app.telemetry import log`; never log document text)
  Grafana (recommended, not graded): src/app/dashboards/outline.json with a panel for outline.calls by status and a logs panel
          ({service_name="extraction-agent"} | json | event="document_outline_done").

---------------------------------------------------------------------------------- task 8.1  delete_document(doc_name: str) -> str
  Removes one document from the user's knowledge base. doc_name is a file name from the KNOWLEDGE BASE list.
  Resolve it with kb.resolve_doc_name(user_id, doc_name); find the document's doc_id in kb.documents(user_id);
  delete with kb.delete_document(user_id, doc_id).
  output  "Deleted '<exact file name>'. <N> documents left."
  error   no such document in THIS user's knowledge base  ->  ERROR[UNKNOWN_DOCUMENT]  (list the valid names in the message)

---------------------------------------------------------------------------------- task 8.2  update_document(doc_name: str, source: str) -> str
  Replaces a document with a new version the user uploaded. source = the file_id of the NEW upload (like ingest_document).
  Order matters: ingest the new file FIRST under the OLD file name (kb/pipeline.ingest_file, files.resolve), and only when that
  succeeded delete the old version by its doc_id: a failed update must leave the old document untouched.
  If the new file is identical to a document already in the knowledge base ingest_file returns it as `existing`: change nothing.
  output  "Updated '<exact file name>': <S> sections, <C> chunks."      identical content: "'<name>' is already up to date."
  errors  unknown doc_name -> ERROR[UNKNOWN_DOCUMENT];  bad/unknown file_id -> use app.errors.to_tool_exception on the DocError
          (it yields ERROR[FILE_NOT_FOUND]).
"""
import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log

_meter = metrics.get_meter("agent")
KB_STATS_CALLS = _meter.create_counter("kbstats.calls", unit="1", description="kb_stats tool calls")
OUTLINE_CALLS = _meter.create_counter("outline.calls", unit="1", description="document_outline tool calls")


def build_custom_tools(kb: KnowledgeBase) -> list:

    @tool
    async def kb_stats(top_keywords: int = 5, runtime: ToolRuntime[Ctx] = None) -> str:
        """Return statistics about the user's knowledge base: document count, section count, chunk count,
        and the most common keywords. Use this when the user asks what is in their knowledge base,
        how many documents they have, or what topics are covered.

        Args:
            top_keywords: how many top keywords to show (1-20, default 5)
        """
        t0 = time.perf_counter()
        status = "error"
        try:
            if not (1 <= top_keywords <= 20):
                raise ToolException(
                    f"ERROR[INVALID_INPUT]: top_keywords must be between 1 and 20, got {top_keywords}. "
                    "Next step: retry with a value in that range."
                )
            user_id = runtime.context.user_id if runtime and runtime.context.user_id else (
                runtime.context.session_id if runtime else "")
            stats = await asyncio.to_thread(kb.stats, user_id)
            if stats["documents"] == 0:
                status = "ok"
                return "The knowledge base is empty."
            kws = stats["keywords"][:top_keywords]
            result = (
                f"{stats['documents']} documents, {stats['sections']} sections, {stats['chunks']} chunks."
                + (f" Top keywords: {', '.join(kws)}." if kws else "")
            )
            status = "ok"
            return result
        except ToolException:
            raise
        except Exception as exc:
            raise to_tool_exception(exc) from exc
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            KB_STATS_CALLS.add(1, {"status": status})
            log.info("kb_stats_done", tool="kb_stats", status=status, duration_ms=elapsed_ms)

    kb_stats.handle_tool_error = True

    @tool
    async def document_outline(doc_name: str, runtime: ToolRuntime[Ctx] = None) -> str:
        """Return the section-by-section structure of one document in the user's knowledge base.
        Use this when the user asks what a document contains, how it is organised, or which section
        covers a particular topic (then search that section). Pass the exact file name from the
        KNOWLEDGE BASE list as doc_name.

        Args:
            doc_name: the file name of the document to outline (from the KNOWLEDGE BASE list)
        """
        t0 = time.perf_counter()
        status = "error"
        n_sections = 0
        try:
            user_id = runtime.context.user_id if runtime and runtime.context.user_id else (
                runtime.context.session_id if runtime else "")
            exact = kb.resolve_doc_name(user_id, doc_name)
            if exact is None:
                names = ", ".join(f'"{d["filename"]}"' for d in kb.documents(user_id)) or "(none)"
                raise ToolException(
                    f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. "
                    f"Valid names: {names}. Next step: retry with one of these names."
                )
            # Find doc_id from documents list
            docs = kb.documents(user_id)
            doc_id = next((d["doc_id"] for d in docs if d["filename"] == exact), None)
            if doc_id is None:
                names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
                raise ToolException(
                    f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. "
                    f"Valid names: {names}. Next step: retry with one of these names."
                )
            sections = await asyncio.to_thread(kb.outline, user_id, doc_id)
            n_sections = len(sections)
            lines = [f"Outline of '{exact}' ({n_sections} sections):"]
            for i, s in enumerate(sections, start=1):
                level = s.get("level", 1)
                heading = s.get("heading", "")
                page_start = s.get("page_start")
                page_end = s.get("page_end")
                summary = s.get("summary", "")
                if page_start is not None:
                    if page_start == page_end:
                        page_part = f" (p.{page_start})"
                    else:
                        page_part = f" (p.{page_start}-{page_end})"
                else:
                    page_part = ""
                lines.append(f"{i}. [L{level}] {heading}{page_part}: {summary}")
            status = "ok"
            return "\n".join(lines)
        except ToolException:
            raise
        except Exception as exc:
            raise to_tool_exception(exc) from exc
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            OUTLINE_CALLS.add(1, {"status": status})
            log.info("document_outline_done", tool="document_outline", status=status,
                     duration_ms=elapsed_ms, sections=n_sections)

    document_outline.handle_tool_error = True

    @tool
    async def delete_document(doc_name: str, runtime: ToolRuntime[Ctx] = None) -> str:
        """Remove one document from the user's knowledge base. After deletion, the agent will no longer
        be able to search or cite that document. Use this when the user explicitly asks to remove a document.

        Args:
            doc_name: the file name of the document to delete (from the KNOWLEDGE BASE list)
        """
        user_id = runtime.context.user_id if runtime and runtime.context.user_id else (
            runtime.context.session_id if runtime else "")
        exact = kb.resolve_doc_name(user_id, doc_name)
        if exact is None:
            names = ", ".join(f'"{d["filename"]}"' for d in kb.documents(user_id)) or "(none)"
            raise ToolException(
                f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. "
                f"Valid names: {names}. Next step: retry with one of these names."
            )
        docs = kb.documents(user_id)
        doc_id = next((d["doc_id"] for d in docs if d["filename"] == exact), None)
        if doc_id is None:
            names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
            raise ToolException(
                f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. "
                f"Valid names: {names}. Next step: retry with one of these names."
            )
        await asyncio.to_thread(kb.delete_document, user_id, doc_id)
        remaining = kb.documents(user_id)
        return f"Deleted '{exact}'. {len(remaining)} documents left."

    delete_document.handle_tool_error = True

    @tool
    async def update_document(doc_name: str, source: str, runtime: ToolRuntime[Ctx] = None) -> str:
        """Replace a document in the user's knowledge base with a newly uploaded version.
        The old document is only removed after the new one is successfully ingested, so a failure
        leaves the old document intact. Use this when the user uploads a new version of an existing document.

        Args:
            doc_name: the file name of the document to replace (from the KNOWLEDGE BASE list)
            source: the file_id of the newly uploaded file (from the upload message)
        """
        user_id = runtime.context.user_id if runtime and runtime.context.user_id else (
            runtime.context.session_id if runtime else "")
        session_id = runtime.context.session_id if runtime else user_id

        # Find the existing document
        exact = kb.resolve_doc_name(user_id, doc_name)
        if exact is None:
            names = ", ".join(f'"{d["filename"]}"' for d in kb.documents(user_id)) or "(none)"
            raise ToolException(
                f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. "
                f"Valid names: {names}. Next step: retry with one of these names."
            )
        docs = kb.documents(user_id)
        old_doc = next((d for d in docs if d["filename"] == exact), None)
        if old_doc is None:
            names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
            raise ToolException(
                f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. "
                f"Valid names: {names}. Next step: retry with one of these names."
            )
        old_doc_id = old_doc["doc_id"]

        # Resolve the new file
        try:
            path, _ = files.resolve(session_id, source.strip())
        except Exception as exc:
            raise to_tool_exception(exc) from exc

        # Ingest new file under old name first
        try:
            result, existing = await ingest_file(
                kb, user_id, path, exact, runtime.config if runtime else None
            )
        except Exception as exc:
            raise to_tool_exception(exc) from exc

        if existing:
            # Identical content - no change needed
            return f"'{exact}' is already up to date."

        # Success - now delete old document
        await asyncio.to_thread(kb.delete_document, user_id, old_doc_id)
        return f"Updated '{exact}': {len(result.sections)} sections, {result.n_chunks} chunks."

    update_document.handle_tool_error = True

    return [kb_stats, document_outline, delete_document, update_document]
