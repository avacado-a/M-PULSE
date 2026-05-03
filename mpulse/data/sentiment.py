import re

class LexiconSentiment:
    """
    Lightweight Lexicon-Based Sentiment Extractor.
    Designed to process short-form, volatile social media text without the overhead of an LLM.
    """
    def __init__(self):
        # A basic dictionary mapping to extract polarity from tech/political domains
        self.positive_lexicon = {
            'good', 'great', 'excellent', 'positive', 'breakthrough', 'success', 'growth', 
            'innovative', 'support', 'agree', 'win', 'improve', 'best', 'better', 'fast',
            'upgrade', 'new', 'stable', 'safe', 'profit', 'rise', 'up', 'bull', 'buy'
        }
        self.negative_lexicon = {
            'bad', 'terrible', 'error', 'fail', 'negative', 'conflict', 'war', 'crisis', 
            'bias', 'decline', 'worse', 'bug', 'crash', 'down', 'loss', 'slow', 'hate',
            'scam', 'sell', 'bear', 'issue', 'problem', 'risk', 'danger', 'attack'
        }

    def score_text(self, text: str) -> float:
        """Returns a normalized sentiment score between -1.0 and 1.0"""
        if not text or not isinstance(text, str):
            return 0.0
            
        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in self.positive_lexicon)
        neg_count = sum(1 for w in words if w in self.negative_lexicon)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
        return (pos_count - neg_count) / total

    def score_daily_aggregate(self, texts: list) -> float:
        """Averages the sentiment over a list of daily posts."""
        if not texts:
            return 0.0
        scores = [self.score_text(t) for t in texts]
        return sum(scores) / len(scores)
