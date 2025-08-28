"""
Flask routes for the Tarot application
"""
import os
from flask import jsonify, request, render_template
from dotenv import load_dotenv
from . import app
from .tarot_agent.agent import TarotAgent

# Load environment variables
load_dotenv()

# Initialize Tarot Agent
agent = TarotAgent(
    cards_path=os.getenv('CARDS_DATA_PATH', './static/images/cards'),
    vector_store_path=os.getenv('VECTOR_STORE_PATH', './vector_store')
)

def init_app(app):
    """Initialize the application"""
    with app.app_context():
        agent.initialize_vector_store()

# Initialize vector store
init_app(app)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/reading', methods=['POST'])
async def get_reading():
    """
    Get a tarot reading
    
    Expected JSON body:
    {
        "question": "User's question"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        # Отримуємо читання з автоматично вибраними картами
        result = await agent.get_reading(
            question=data['question']
        )
        
        return jsonify({
            'question': data['question'],
            'cards': result['cards'],
            'reading': result['reading']
        })
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in get_reading: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/card/<card_name>', methods=['GET'])
def get_card_info(card_name):
    """Get information about a specific card"""
    try:
        card_info = agent.get_card_info(card_name)
        if card_info:
            return jsonify(card_info)
        return jsonify({'error': 'Card not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/guardrails/stats', methods=['GET'])
def get_guardrails_stats():
    """Get Guardrails statistics"""
    try:
        if hasattr(agent, 'guardrails'):
            stats = agent.guardrails.get_stats()
            return jsonify(stats)
        return jsonify({'error': 'Guardrails not initialized'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/guardrails/reset', methods=['POST'])
def reset_guardrails_stats():
    """Reset Guardrails statistics"""
    try:
        if hasattr(agent, 'guardrails'):
            agent.guardrails.reset_stats()
            return jsonify({'message': 'Statistics reset successfully'})
        return jsonify({'error': 'Guardrails not initialized'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        stats = {}
        if hasattr(agent, 'guardrails'):
            stats = agent.guardrails.get_stats()
        
        return jsonify({
            'status': 'healthy',
            'vector_store_initialized': agent.vector_store is not None,
            'guardrails_stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500