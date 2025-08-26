"""
Module for loading and preparing tarot card data for the vector store
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TarotCard:
    """Data class for storing tarot card information"""
    name: str
    description: str
    upright_meaning: str
    reversed_meaning: str
    short_upright: str
    short_reversed: str
    card_type: str  # 'major' or 'minor'
    suit: Optional[str] = None  # For minor arcana only

class TarotDataLoader:
    """Class for loading and preparing tarot card data"""
    
    def __init__(self, base_path: str):
        """Initialize with base path to card data"""
        self.base_path = base_path
        self.major_arcana_path = os.path.join(base_path, 'MajorArcana')
        self.minor_arcana_paths = {
            'Cups': os.path.join(base_path, 'MinorArcana_Cups'),
            'Pentacles': os.path.join(base_path, 'MinorArcana_Pentacles'),
            'Swords': os.path.join(base_path, 'MinorArcana_Swords'),
            'Wands': os.path.join(base_path, 'MinorArcana_Wands')
        }

    def _read_file_content(self, filepath: str) -> str:
        """Read content from a file, return empty string if file doesn't exist"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def _load_card_data(self, card_number: int, card_path: str, 
                       card_type: str, suit: Optional[str] = None) -> TarotCard:
        """Load all data for a single card"""
        base_name = str(card_number)
        return TarotCard(
            name=self._read_file_content(os.path.join(card_path, f'{base_name}-name.txt')),
            description=self._read_file_content(os.path.join(card_path, f'{base_name}-desc.txt')),
            upright_meaning=self._read_file_content(os.path.join(card_path, f'{base_name}-umean.txt')),
            reversed_meaning=self._read_file_content(os.path.join(card_path, f'{base_name}-rmean.txt')),
            short_upright=self._read_file_content(os.path.join(card_path, f'{base_name}-short-u.txt')),
            short_reversed=self._read_file_content(os.path.join(card_path, f'{base_name}-short-r.txt')),
            card_type=card_type,
            suit=suit
        )

    def load_all_cards(self) -> List[TarotCard]:
        """Load all tarot cards data"""
        cards = []
        
        # Load Major Arcana
        for i in range(22):  # 0-21 for major arcana
            cards.append(self._load_card_data(i, self.major_arcana_path, 'major'))
        
        # Load Minor Arcana
        for suit, path in self.minor_arcana_paths.items():
            for i in range(14):  # 0-13 for minor arcana
                cards.append(self._load_card_data(i, path, 'minor', suit))
        
        return cards

    def prepare_documents(self) -> List[Dict[str, str]]:
        """Prepare documents for vector store"""
        cards = self.load_all_cards()
        documents = []
        
        for card in cards:
            # Create comprehensive document for each card
            doc = {
                'content': f"""
                Card: {card.name}
                Type: {card.card_type.title()} Arcana
                {f'Suit: {card.suit}' if card.suit else ''}
                
                Description:
                {card.description}
                
                Upright Meaning:
                {card.upright_meaning}
                Key upright meanings: {card.short_upright}
                
                Reversed Meaning:
                {card.reversed_meaning}
                Key reversed meanings: {card.short_reversed}
                """.strip(),
                'metadata': {
                    'name': card.name,
                    'type': card.card_type,
                    'suit': card.suit if card.suit else 'NA'
                }
            }
            documents.append(doc)
            
            # Create additional documents for specific aspects
            aspects = [
                ('upright', card.upright_meaning, card.short_upright),
                ('reversed', card.reversed_meaning, card.short_reversed)
            ]
            
            for aspect, meaning, short in aspects:
                doc = {
                    'content': f"""
                    Card: {card.name} ({aspect})
                    Type: {card.card_type.title()} Arcana
                    {f'Suit: {card.suit}' if card.suit else ''}
                    
                    {meaning}
                    
                    Key meanings: {short}
                    """.strip(),
                    'metadata': {
                        'name': card.name,
                        'type': card.card_type,
                        'suit': card.suit if card.suit else 'NA',
                        'aspect': aspect
                    }
                }
                documents.append(doc)
        
        return documents
