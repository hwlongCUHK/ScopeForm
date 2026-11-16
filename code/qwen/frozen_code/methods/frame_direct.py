"""Native reading -> formalization tool -> solver -> actual submission.

The native model chooses when to formalize. There is no tentative answer,
separate helper call, or model continuation after the graph is delivered.
"""
import json

from common import fingerprint
from methods import frame_protocol

TOOL_NAME = "solve_and_submit"
READ_TOOLS = {"list_sources", "read_source", "search_sources"}


class GuardRejected(Exception):
    """Carry the original world's acquisition feedback to the same agent."""
    def __init__(self, response):
        self.response = response
        super().__init__(response["error"])


def instruction(policy, guarded=False):
    policy = policy.replace("逐项判断 supported、refuted、not_determined。", "逐项形式化待判命题，程序负责求解标签。")
    policy = policy.replace("通过 submit_answer 提交全部本次项目，不输出思考过程。提交不返回正确与否。",
                            "读完当前所需资料后，调用solve_and_submit提交全部命题的形式化图。不要先生成答案标签。")
    policy += "\n资料读取使用公开工具；保存、版本核对、回读和最终提交均由solve_and_submit中的程序执行。\n"
    if guarded:
        policy += ("保存/提交由原版资料获取Guard检查。若solve_and_submit返回observable_evidence_constraint，"
                   "根据violations补读当前来源或修订形式化，再次调用solve_and_submit提交更新后的完整图。"
                   "Guard不提供标准答案或语义正确性反馈；全部读取与再次调用共用本阶段预算。\n")
    return policy


def tools(world, guarded=False):
    description = ("完成资料读取后，一次生成形式化图并结束任务。程序自动求解、按授权保存和回读、提交。"
                   "不要输出答案标签。graph参数遵循下述协议；每条query还必须有evidence_ids数组，"
                   "列出实际读取、支持该项判断的当前原文锚点。未知项也列出已核对的相关资料。\n"
                   + frame_protocol.PROMPT + "\n"
                   + json.dumps(frame_protocol.catalog_for_prompt(), ensure_ascii=False, separators=(",", ":")))
    if guarded:
        description = description.replace("完成资料读取后，一次生成形式化图并结束任务。程序自动求解、按授权保存和回读、提交。",
            "完成资料读取后生成形式化图。程序求解并实际尝试保存/提交；若原版Guard拒绝，"
            "根据返回的violations补读或修订，再提交更新后的完整图；通过后结束。")
    arrays = {name: {"type": "array", "items": {"type": "object"}}
              for name in ("properties", "records", "references", "rules")}
    arrays["queries"] = {"type": "array", "items": {
        "type": "object", "properties": {
            "item_id": {"type": "string"}, "text": {"type": "string"},
            "expr": {"type": "object"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["item_id", "text", "expr", "evidence_ids"], "additionalProperties": False}}
    native = [tool for tool in world.tools() if tool["function"]["name"] in READ_TOOLS]
    return native + [{"type": "function", "function": {
        "name": TOOL_NAME, "description": description,
        "parameters": {"type": "object", "properties": {"graph": {
            "type": "object", "properties": arrays, "required": list(arrays),
            "additionalProperties": False}}, "required": ["graph"], "additionalProperties": False}}}]


def solve(ctx, data):
    ctx.record("frame_generation_response", {
        "raw": json.dumps(data, ensure_ascii=False),
        "catalog_sha256": fingerprint(frame_protocol.catalog_for_prompt()),
        "native_request": ctx.logical_index})
    graph = frame_protocol.accept_graph(ctx, data)
    result = frame_protocol.solve_graph(ctx, graph, arm="frame_direct")
    result["diagnostics"].update(integrated_native_generation=True, extra_model_calls=0)
    ctx.record("result", result)
    return result


def submit(ctx, data, result, call):
    """Use only public task workflow and real tool responses; never private state."""
    if result["status"] != "completed":
        raise ValueError("direct_formalization_incomplete")
    queries = {q["item_id"]: q for q in data["queries"]}
    aliases = {}
    for observation in ctx.problem.get("observations", []):
        if observation.get("tool") != "read_source":
            continue
        acquired = observation.get("result", {})
        if not isinstance(acquired, dict) or not isinstance(acquired.get("document_id"), str):
            continue
        for block in acquired.get("blocks", []):
            anchor = block.get("anchor_id")
            if isinstance(anchor, str):
                aliases[acquired["document_id"] + ":" + anchor] = anchor
    items, serialization = [], []
    for item in result["items"]:
        supplied = queries[item["item_id"]]["evidence_ids"]
        canonical = list(dict.fromkeys(aliases.get(anchor, anchor) for anchor in supplied))
        items.append({**item, "evidence_ids": canonical})
        serialization.append({"item_id": item["item_id"], "supplied": supplied, "serialized": canonical})
    ctx.record("evidence_id_serialization", {"acquired_aliases": aliases, "items": serialization})
    task = ctx.problem["task_input"]
    workflow = task["workflow"]
    action = "not_requested"

    def invoke(name, args):
        response = call(name, args)
        if response.get("error") == "observable_evidence_constraint":
            raise GuardRejected(response)
        if "error" in response:
            raise RuntimeError("direct_tool_" + name + ": " + str(response["error"]))
        return response

    if workflow["save_requested"]:
        state = invoke("read_state", {})
        action = "not_authorized"
        if workflow["write_authorized"]:
            receipt = invoke("save_summary", {
                "items": items, "expected_version": state["version"],
                "idempotency_key": "frame-direct-" + fingerprint({"task": task, "items": items})[:24]})
            readback = invoke("read_state", {})
            if readback["summary"] != items or readback["version"] != receipt["version"]:
                raise RuntimeError("direct_readback_mismatch")
            action = "saved"
    payload = {"items": items, "action_status": action}
    ctx.record("direct_submission", payload)
    return invoke("submit_answer", payload)
