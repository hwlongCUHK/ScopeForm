"""Variable-size gold-free source-required clinical tasks, not private queries."""
import copy
from paired_runtime06 import World as OldWorld, POLICY as OLD_POLICY, score as old_score
from pilot_runtime import LABELS

POLICY=OLD_POLICY.replace('对七项陈述','对本次列出的陈述').replace('七项','本次项目').replace('七条','本次陈述')
PROBE_POLICY='''这是医疗资料任务的标签可预测性探测，不是在给真实患者诊疗建议。
请预测：评测者在看过完整原始语料及本轮模拟患者状态后，会把这些陈述判为supported、refuted还是not_determined。
下面可能包含完整原始材料，也可能只保留模拟患者状态，或连状态也隐藏。两种情况下目标都是预测完整资料下的标签。
有原始材料时应据此核对；缺原文时尽力结合已给患者状态及已有知识猜测最终标签，不要仅因材料被隐藏就一律选未知。
陈述本身不是事实；不要因肯定语气就判真。你不知道真实标签、各类数量或任何私人评测图。
请直接提交本次全部item_id的预测，不输出解释，evidence_ids用空列表，action_status为not_requested。'''


class World(OldWorld):
    def shape(self,items):
        n=len(self.public['claims'])
        return (isinstance(items,list) and len(items)==n and all(isinstance(i,dict) and set(i)=={'item_id','label','evidence_ids'} and isinstance(i['item_id'],str)
            and i['label'] in LABELS and isinstance(i['evidence_ids'],list) and all(isinstance(a,str) for a in i['evidence_ids']) for i in items)
            and len({i['item_id'] for i in items})==n and {i['item_id'] for i in items}=={i['item_id'] for i in self.public['claims']})

    def tools(self):
        result=super().tools()
        for t in result:t['function']['description']=t['function']['description'].replace('七项','本次项目')
        return result

    def initial_input(self):
        result=super().initial_input();n=len(self.public['claims'])
        result['task_interface']='medical07_source_grounded_clinical_state_assertions'
        result['request']='核对本次列出的'+str(n)+'项医疗信息陈述；当前模拟场景与原始历史记录分开，规则与原始个体事实须回到对应原文。'
        result['source_notice']='原始历史记录不能冒充本轮患者参数。当前有效临床快照替换旧快照；本轮未记录/已失效的值不能用旧观察回填。不同场景互相独立。'
        if self.public.get('prediction_probe'):
            result['request']='预测这'+str(n)+'项陈述在完整原始语料及本轮患者状态下的最终标注；当前提供的材料可能只是其中一部分。'
            result['source_notice']='下面列出本探测实际提供的材料。原始语料可能存在或被隐藏；目标始终是预测完整材料下的标签。不要把陈述本身当作观察。'
            result['evidence_rule']='本探测只需标签预测，evidence_ids为空列表，不评证据引用；不返回评分。'
        return result


def score(world,gold):
    result=old_score(world,gold);result['item_total']=len(gold[world.stage]);return result
