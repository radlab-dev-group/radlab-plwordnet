import re

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


class ExampleType(Enum):
    """Enum for different types of usage examples"""

    STANDARD = "P"
    KPWR = "KPWr"
    UNKNOWN = "UNKNOWN"


@dataclass
class SentimentAnnotation:
    """
    Data class representing a sentiment annotation (##A1, ##A2, etc.)
    """

    annotation_id: str  # e.g., "A1", "A2"
    emotions: List[str]  # e.g., ["smutek", "złość"]
    categories: List[str]  # e.g., ["błąd"]
    strength: str  # e.g., "s" for strong, "w" for weak, etc.
    example: str  # example sentence in brackets


@dataclass
class UsageExample:
    """
    Data class representing a usage example with type information
    """

    text: str
    example_type: ExampleType
    source_pattern: Optional[str] = None


@dataclass
class ExternalUrlDescription:
    """
    Data class representing an external URL description (##L)
    """

    url: str
    content: str = None


@dataclass
class ParsedComment:
    """
    Data class representing the parsed comment structure
    """

    original_comment: str
    base_domain: Optional[str] = None  # ##K content
    definition: Optional[str] = None  # ##D content
    usage_examples: List[UsageExample] = None  # All examples from [ ]
    external_url_description: Optional[ExternalUrlDescription] = None  # ##L content
    sentiment_annotations: List[SentimentAnnotation] = None  # ##A1, ##A2, etc.

    def __post_init__(self):
        if self.usage_examples is None:
            self.usage_examples = []
        if self.sentiment_annotations is None:
            self.sentiment_annotations = []


class CommentParser:
    """
    Parser for plWordnet comment annotations with sentiment analysis.
    """

    def __init__(self):
        # Regex patterns for different comment elements
        self.base_domain_pattern = r"##K:\s*([^#]+?)(?=\s*##|$)"
        self.definition_pattern = r"##D:\s*([^#\[{]+?)(?=\s*\[|##|{|$)"
        self.external_url_pattern = r"{##L:\s*([^}]+?)}"

        # Updated sentiment annotation pattern to capture strength
        self.sentiment_annotation_pattern = (
            r"##(A\d+):\s*\{([^}]+)\}\s*-\s*(\w+)\s*\[([^\]]+)\]"
        )

        # Pattern to find all bracketed examples (excluding sentiment annotations)
        self.bracketed_example_pattern = r"\[([^\]]+?)\]"

        # Pattern to identify the example type within brackets
        self.example_type_pattern = r"##([A-Za-z0-9]+):\s*(.+)"

    def parse_comment(self, comment: str) -> ParsedComment:
        """
        Parse a plWordnet comment string into structured data.

        Args:
            comment: Raw comment string from database

        Returns:
            ParsedComment: Parsed comment structure
        """
        if not comment or not comment.strip():
            return ParsedComment(original_comment=comment)

        parsed = ParsedComment(original_comment=comment)
        parsed.base_domain = self._extract_base_domain(comment=comment)
        parsed.definition = self._extract_definition(comment=comment)
        parsed.sentiment_annotations = self._extract_sentiment_annotations(
            comment=comment
        )

        parsed.usage_examples = self._extract_bracketed_examples(
            comment=comment, sentiment_annotations=parsed.sentiment_annotations
        )

        parsed.external_url_description = self._extract_external_url_description(
            comment=comment
        )

        return parsed

    def get_examples_by_type(
        self, parsed_comment: ParsedComment
    ) -> Dict[str, List[str]]:
        """
        Get usage examples grouped by type.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            Dictionary with usage examples grouped by type
        """
        examples_by_type = {
            "standard": [],
            "kpwr": [],
            "unknown": [],
            "sentiment": [],
        }

        # Group regular usage examples
        for example in parsed_comment.usage_examples:
            if example.example_type == ExampleType.STANDARD:
                examples_by_type["standard"].append(example.text)
            elif example.example_type == ExampleType.KPWR:
                examples_by_type["kpwr"].append(example.text)
            else:
                examples_by_type["unknown"].append(example.text)

        # Add sentiment examples
        examples_by_type["sentiment"] = self.get_sentiment_examples(
            parsed_comment=parsed_comment
        )

        return examples_by_type

    def get_comment_statistics(
        self, parsed_comment: ParsedComment
    ) -> Dict[str, int]:
        """
        Get statistics about the parsed comment.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            Dictionary with comment statistics
        """
        examples_by_type = self.get_examples_by_type(parsed_comment=parsed_comment)

        return {
            "total_usage_examples": len(parsed_comment.usage_examples),
            "standard_examples": len(examples_by_type["standard"]),
            "kpwr_examples": len(examples_by_type["kpwr"]),
            "unknown_examples": len(examples_by_type["unknown"]),
            "sentiment_annotations_count": len(parsed_comment.sentiment_annotations),
            "sentiment_examples": len(examples_by_type["sentiment"]),
            "unique_strengths": len(self.get_all_strengths(parsed_comment)),
            "has_definition": 1 if parsed_comment.definition else 0,
            "has_base_domain": 1 if parsed_comment.base_domain else 0,
            "has_external_url": 1 if parsed_comment.external_url_description else 0,
            "total_examples": len(parsed_comment.usage_examples)
            + len(parsed_comment.sentiment_annotations),
        }

    @staticmethod
    def _parse_emotions_and_categories(content: str) -> Tuple[List[str], List[str]]:
        """
        Parse emotions and categories from the content inside {}.

        Expected format: "emotion1, emotion2; category1, category2"
        """
        emotions = []
        categories = []

        if ";" in content:
            # Split by semicolon to separate emotions from categories
            parts = content.split(";", 1)
            emotions_part = parts[0].strip()
            categories_part = parts[1].strip() if len(parts) > 1 else ""

            # Parse emotions
            if emotions_part:
                emotions = [emotion.strip() for emotion in emotions_part.split(",")]

            # Parse categories
            if categories_part:
                categories = [
                    category.strip() for category in categories_part.split(",")
                ]
        else:
            # If no semicolon, treat everything as emotions
            emotions = [emotion.strip() for emotion in content.split(",")]

        return emotions, categories

    @staticmethod
    def get_all_emotions(parsed_comment: ParsedComment) -> List[str]:
        """
        Get all unique emotions from all sentiment annotations.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            List of unique emotions
        """
        all_emotions = []
        for annotation in parsed_comment.sentiment_annotations:
            all_emotions.extend(annotation.emotions)
        return list(set(all_emotions))

    @staticmethod
    def get_all_categories(parsed_comment: ParsedComment) -> List[str]:
        """
        Get all unique categories from all sentiment annotations.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            List of unique categories
        """
        all_categories = []
        for annotation in parsed_comment.sentiment_annotations:
            all_categories.extend(annotation.categories)
        return list(set(all_categories))

    @staticmethod
    def get_all_strengths(parsed_comment: ParsedComment) -> List[str]:
        """
        Get all unique strengths from all sentiment annotations.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            List of unique strengths
        """
        return list(
            set(
                annotation.strength
                for annotation in parsed_comment.sentiment_annotations
            )
        )

    @staticmethod
    def has_sentiment_annotation(parsed_comment: ParsedComment) -> bool:
        """
        Check if the comment has any sentiment annotations.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            True if it has sentiment annotations, False otherwise
        """
        return len(parsed_comment.sentiment_annotations) > 0

    @staticmethod
    def get_sentiment_examples(parsed_comment: ParsedComment) -> List[str]:
        """
        Get all sentiment annotation examples.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            List of sentiment annotation examples
        """
        return [
            annotation.example for annotation in parsed_comment.sentiment_annotations
        ]

    @staticmethod
    def get_examples_by_source_pattern(
        parsed_comment: ParsedComment,
    ) -> Dict[str, List[str]]:
        """
        Get usage examples grouped by their source pattern.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            Dictionary with usage examples grouped by source pattern
        """
        examples_by_pattern = {}
        for example in parsed_comment.usage_examples:
            pattern = example.source_pattern or "NO_PATTERN"
            if pattern not in examples_by_pattern:
                examples_by_pattern[pattern] = []
            examples_by_pattern[pattern].append(example.text)
        # Add sentiment annotations with their IDs as patterns
        for annotation in parsed_comment.sentiment_annotations:
            pattern = f"##A{annotation.annotation_id}"
            if pattern not in examples_by_pattern:
                examples_by_pattern[pattern] = []
            examples_by_pattern[pattern].append(annotation.example)
        return examples_by_pattern

    @staticmethod
    def get_sentiment_annotations_by_strength(
        parsed_comment: ParsedComment,
    ) -> Dict[str, List[SentimentAnnotation]]:
        """
        Get sentiment annotations grouped by strength.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            Dictionary with sentiment annotations grouped by strength
        """
        annotations_by_strength = {}
        for annotation in parsed_comment.sentiment_annotations:
            strength = annotation.strength
            if strength not in annotations_by_strength:
                annotations_by_strength[strength] = []
            annotations_by_strength[strength].append(annotation)
        return annotations_by_strength

    @staticmethod
    def has_external_url(parsed_comment: ParsedComment) -> bool:
        """
        Check if the comment has an external URL.

        Args:
            parsed_comment: Parsed comment object

        Returns:
            True if it has external URL, False otherwise
        """
        return parsed_comment.external_url_description is not None

    def _extract_base_domain(self, comment: str) -> Optional[str]:
        """Extract base_domain from ##K tag."""
        match = re.search(self.base_domain_pattern, comment)
        if match:
            return match.group(1).strip()
        return None

    def _extract_definition(self, comment: str) -> Optional[str]:
        """Extract definition from ##D tag."""
        match = re.search(self.definition_pattern, comment)
        if match:
            return match.group(1).strip()
        return None

    def _extract_sentiment_annotations(
        self, comment: str
    ) -> List[SentimentAnnotation]:
        """
        Extract sentiment annotations from ##A1, ##A2, etc. tags.
        Format: ##A1: {emotions; categories} - strength [example]
        """
        annotations = []
        matches = re.findall(self.sentiment_annotation_pattern, comment)

        for match in matches:
            annotation_id = match[0]  # A1, A2, etc.
            emotions_and_categories = match[1]  # content inside {}
            strength = match[2]  # s, w, etc.
            example = match[3]  # content inside []
            # Parse emotions and categories from the {} content
            emotions, categories = self._parse_emotions_and_categories(
                content=emotions_and_categories
            )

            annotation = SentimentAnnotation(
                annotation_id=annotation_id,
                emotions=emotions,
                categories=categories,
                strength=strength,
                example=example.strip(),
            )
            annotations.append(annotation)

        return annotations

    def _extract_bracketed_examples(
        self, comment: str, sentiment_annotations: List[SentimentAnnotation]
    ) -> List[UsageExample]:
        """
        Extract usage examples from [ ] brackets, excluding those already captured in sentiment annotations.

        Args:
            comment: Raw comment string
            sentiment_annotations: Already parsed sentiment annotations to exclude

        Returns:
            List of UsageExample objects with type information
        """
        examples = []
        sentiment_example_texts = {
            annotation.example for annotation in sentiment_annotations
        }
        bracketed_matches = re.findall(self.bracketed_example_pattern, comment)
        for bracketed_content in bracketed_matches:
            # Skip if this is a sentiment annotation example
            if bracketed_content.strip() in sentiment_example_texts:
                continue

            example_type, source_pattern, example_text = (
                self._determine_example_type(bracketed_content=bracketed_content)
            )
            example = UsageExample(
                text=example_text,
                example_type=example_type,
                source_pattern=source_pattern,
            )
            examples.append(example)
        return examples

    def _determine_example_type(
        self, bracketed_content: str
    ) -> Tuple[ExampleType, Optional[str], str]:
        """
        Determine the type of example based on the ##STR: pattern inside brackets.

        Args:
            bracketed_content: Content inside [ ] brackets

        Returns:
            Tuple of (ExampleType, source_pattern, actual_example_text)
        """
        match = re.match(self.example_type_pattern, bracketed_content)

        if match:
            pattern_str = match.group(1)
            example_text = match.group(2).strip()

            if pattern_str == "P":
                return ExampleType.STANDARD, f"##{pattern_str}", example_text
            elif pattern_str == "KPWr":
                return ExampleType.KPWR, f"##{pattern_str}", example_text
            else:
                return ExampleType.UNKNOWN, f"##{pattern_str}", example_text
        else:
            return ExampleType.UNKNOWN, None, bracketed_content.strip()

    def _extract_external_url_description(
        self, comment: str
    ) -> Optional[ExternalUrlDescription]:
        """Extract external URL description from ##L tag."""
        match = re.search(self.external_url_pattern, comment)
        if match:
            url = match.group(1).strip()
            return ExternalUrlDescription(url=url)
        return None


# Utility function for easy parsing
def parse_plwordnet_comment(comment: str) -> ParsedComment:
    """
    Convenience function to parse a plWordnet comment.

    Args:
        comment: Raw comment string

    Returns:
        ParsedComment: Parsed comment structure
    """
    parser = CommentParser()
    return parser.parse_comment(comment)
