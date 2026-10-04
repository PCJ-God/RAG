"""AI 安全护栏：接入阿里云内容安全（绿网）文本审核增强版 PLUS。

- 输入护栏：`llm_query_moderation` —— 检测用户提问（限 2000 字）
- 输出护栏：`llm_response_moderation` —— 检测模型回答（限 5000 字）

未配置 RAM 用户 AccessKey 时自动禁用（跳过检测）；检测调用异常时 fail-open
（放行并告警），避免护栏故障拖垮整个问答链路。
"""
import json
import uuid
from typing import List, Optional

from src.config import Config, settings

BLOCKED_REPLY = "抱歉，这个问题我暂时无法回答，我们换个话题继续学习 RAG 知识吧。"


class GuardResult:
    """一次内容检测的结果。"""

    def __init__(
        self,
        passed: bool,
        risk_level: str = "none",
        labels: Optional[List[str]] = None,
    ):
        self.passed = passed
        self.risk_level = risk_level
        self.labels = labels or []


class ContentGuard:
    """文本内容安全护栏。"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or settings
        self._client = None
        self._block_levels = {
            lv.strip().lower()
            for lv in self.config.guard_block_levels.split(",")
            if lv.strip()
        }
        self.enabled = bool(
            self.config.aliyun_access_key_id
            and self.config.aliyun_access_key_secret
        )

    # ---------- 对外接口 ----------
    def check_input(self, text: str) -> GuardResult:
        """检测用户提问。"""
        return self._check(text, self.config.guard_query_service)

    def check_output(self, text: str) -> GuardResult:
        """检测模型回答。"""
        return self._check(text, self.config.guard_response_service)

    # ---------- 内部 ----------
    def _get_client(self):
        if self._client is None:
            from alibabacloud_green20220302.client import Client
            from alibabacloud_tea_openapi.models import Config as AliyunConfig

            client_config = AliyunConfig(
                access_key_id=self.config.aliyun_access_key_id,
                access_key_secret=self.config.aliyun_access_key_secret,
                connect_timeout=10000,
                read_timeout=3000,
                region_id=self.config.content_security_region,
                endpoint=self.config.content_security_endpoint,
            )
            self._client = Client(client_config)
        return self._client

    def _check(self, text: str, service: str) -> GuardResult:
        if not self.enabled:
            return GuardResult(passed=True)
        text = (text or "").strip()
        if not text:
            return GuardResult(passed=True)

        from alibabacloud_green20220302 import models

        request = models.TextModerationPlusRequest(
            service=service,
            service_parameters=json.dumps(
                {"content": text, "dataId": str(uuid.uuid1())},
                ensure_ascii=False,
            ),
        )
        try:
            response = self._get_client().text_moderation_plus(request)
        except Exception as e:
            print(f"[safety] 护栏调用失败（放行）: {e}")
            return GuardResult(passed=True)

        status_code = getattr(response, "status_code", 200)
        body = getattr(response, "body", None)
        code = getattr(body, "code", None) if body is not None else None
        data = getattr(body, "data", None) if body is not None else None

        if status_code != 200 or code != 200 or data is None:
            message = getattr(body, "message", "") if body is not None else ""
            print(f"[safety] 护栏返回异常 code={code} message={message}（放行）")
            return GuardResult(passed=True)

        risk_level = (getattr(data, "risk_level", None) or "none").lower()
        labels = [
            item.label
            for item in (getattr(data, "result", None) or [])
            if getattr(item, "label", None)
            and getattr(item, "label", None) != "nonLabel"
        ]

        passed = risk_level not in self._block_levels
        if not passed:
            print(f"[safety] 命中风险 risk_level={risk_level} labels={labels}")
        return GuardResult(passed=passed, risk_level=risk_level, labels=labels)
