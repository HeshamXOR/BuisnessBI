"""
Response Parser Module
======================
Parse and structure LLM text responses into usable formats.
"""

import re
from typing import List, Dict, Optional


class ResponseParser:
    """Parse structured content from LLM text responses."""

    @staticmethod
    def parse_sections(text: str) -> Dict[str, str]:
        """
        Parse markdown-style sections from LLM output.

        Args:
            text: LLM response text with ## headers.

        Returns:
            Dictionary mapping section titles to their content.
        """
        sections = {}
        current_section = "Introduction"
        current_content = []

        for line in text.split("\n"):
            # Match ## headers
            header_match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = header_match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    @staticmethod
    def parse_recommendations(text: str) -> List[Dict[str, str]]:
        """
        Extract structured recommendations from LLM output.

        Returns:
            List of dicts with 'title', 'priority', 'details'.
        """
        recommendations = []
        lines = text.split("\n")

        current_rec = {}
        for line in lines:
            line = line.strip()

            # Match numbered items or bold headers
            num_match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*:?\s*(.*)", line)
            if num_match:
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {
                    "title": num_match.group(1).strip(),
                    "details": num_match.group(2).strip()
                }
                continue

            # Match priority indicators
            priority_match = re.match(
                r".*\*\*Priority\*\*:?\s*(High|Medium|Low)", line, re.IGNORECASE
            )
            if priority_match and current_rec:
                current_rec["priority"] = priority_match.group(1).strip()
                continue

            # Match impact
            impact_match = re.match(r".*\*\*Impact\*\*:?\s*(.+)", line, re.IGNORECASE)
            if impact_match and current_rec:
                current_rec["impact"] = impact_match.group(1).strip()
                continue

            # Append continuation lines
            if current_rec and line.startswith("-"):
                current_rec["details"] = (
                    current_rec.get("details", "") + " " + line.lstrip("- ")
                ).strip()

        if current_rec:
            recommendations.append(current_rec)

        return recommendations

    @staticmethod
    def parse_bullet_points(text: str) -> List[str]:
        """Extract bullet points from text."""
        bullets = []
        for line in text.split("\n"):
            line = line.strip()
            if re.match(r"^[-*•]\s+", line):
                bullet = re.sub(r"^[-*•]\s+", "", line).strip()
                if bullet:
                    bullets.append(bullet)
        return bullets

    @staticmethod
    def parse_key_value_pairs(text: str) -> Dict[str, str]:
        """
        Extract key-value pairs from text like 'Key: Value' or '**Key**: Value'.
        """
        pairs = {}
        for line in text.split("\n"):
            line = line.strip()
            # Match **Key**: Value pattern
            kv_match = re.match(r"\*\*(.+?)\*\*:?\s*(.+)", line)
            if kv_match:
                pairs[kv_match.group(1).strip()] = kv_match.group(2).strip()
                continue
            # Match Key: Value pattern
            kv_match = re.match(r"^([A-Za-z ]+):\s*(.+)", line)
            if kv_match:
                pairs[kv_match.group(1).strip()] = kv_match.group(2).strip()
        return pairs

    @staticmethod
    def extract_insights(text: str) -> List[str]:
        """
        Extract key insights (numbered or bulleted items) from text.
        """
        insights = []

        for line in text.split("\n"):
            line = line.strip()
            # Match numbered items
            num_match = re.match(r"^\d+\.\s+(.+)", line)
            if num_match:
                insight = num_match.group(1).strip()
                # Remove bold markers for clean text
                insight = re.sub(r"\*\*(.+?)\*\*", r"\1", insight)
                insights.append(insight)
                continue

            # Match bullet points with substantive content (>20 chars)
            bullet_match = re.match(r"^[-*•]\s+(.{20,})", line)
            if bullet_match:
                ins = bullet_match.group(1).strip()
                ins = re.sub(r"\*\*(.+?)\*\*", r"\1", ins)
                insights.append(ins)

        return insights

    @staticmethod
    def get_summary(text: str, max_sentences: int = 3) -> str:
        """Get the first N sentences as a summary."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary_sentences = sentences[:max_sentences]
        return " ".join(summary_sentences)

    @staticmethod
    def clean_response(text: str) -> str:
        """Clean up LLM response — remove artifacts, normalize whitespace."""
        # Remove code block markers
        text = re.sub(r"```[\w]*\n?", "", text)
        # Normalize multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    @staticmethod
    def is_low_quality_response(text: str) -> bool:
        """Heuristic check for unstable or off-target model output."""
        if not text:
            return True

        lower_text = text.lower()
        red_flags = [
            "i'm sorry",
            "i apologize",
            "the provided text is too long",
            "the provided document",
            "cannot be completed as requested",
            "answer(slave",
            "phase one",
            "instruction manual",
            "no such package",
            "textbook section",
        ]

        red_flag_hits = sum(lower_text.count(flag) for flag in red_flags)
        has_markdown_structure = "## " in text
        has_numeric_evidence = bool(re.search(r"\d", text))

        if red_flag_hits >= 2:
            return True
        if red_flag_hits >= 1 and not has_numeric_evidence:
            return True
        if len(text) > 600 and not has_markdown_structure:
            return True

        return False
