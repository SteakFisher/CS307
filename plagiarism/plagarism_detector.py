from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

class DocumentAnalyzer:
    """
    Advanced text analysis system for detecting content similarity
    """
    
    def __init__(self):
        self._initialize_nltk_resources()
        self.vectorization_engine = TfidfVectorizer()
    
    def _initialize_nltk_resources(self):
        """Download required NLTK components"""
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    def calculate_textual_similarity(self, first_fragment, second_fragment):
        """
        Measure semantic similarity between two text fragments using TF-IDF vectorization
        """
        if not first_fragment or not second_fragment:
            return 0.0
        
        text_corpus = [first_fragment, second_fragment]
        vectorized_content = self.vectorization_engine.fit_transform(text_corpus)
        similarity_coefficients = cosine_similarity(vectorized_content)
        
        return similarity_coefficients[0, 1]
    
    def fragment_and_normalize_content(self, raw_text):
        """
        Break down text into sentences and normalize them for comparison
        """
        sentence_fragments = nltk.sent_tokenize(raw_text)
        cleaned_fragments = []
        
        for fragment in sentence_fragments:
            # Convert to lowercase and remove punctuation
            normalized_fragment = fragment.lower()
            punctuation_translator = str.maketrans('', '', string.punctuation)
            normalized_fragment = normalized_fragment.translate(punctuation_translator)
            cleaned_fragments.append(normalized_fragment)
        
        return cleaned_fragments
    
    def assess_content_overlap(self, original_fragments, comparison_fragments):
        """
        Evaluate the degree of similarity between two sets of text fragments
        """
        accumulated_similarity = 0.0
        fragment_comparisons = []
        
        for original_piece in original_fragments:
            highest_similarity = 0.0
            closest_match = None
            
            # Find the best matching fragment in comparison text
            for comparison_piece in comparison_fragments:
                current_similarity = self.calculate_textual_similarity(original_piece, comparison_piece)
                if current_similarity > highest_similarity:
                    highest_similarity = current_similarity
                    closest_match = comparison_piece
            
            fragment_comparisons.append({
                'original': original_piece,
                'matched': closest_match,
                'confidence': highest_similarity
            })
            accumulated_similarity += highest_similarity
        
        # Calculate overall similarity percentage
        overall_similarity = (accumulated_similarity / len(original_fragments)) if original_fragments else 0.0
        return overall_similarity, fragment_comparisons

class PlagiarismDetectionSystem:
    """
    Main system for orchestrating plagiarism detection workflow
    """
    
    def __init__(self):
        self.text_analyzer = DocumentAnalyzer()
    
    def execute_similarity_analysis(self, primary_document, secondary_document):
        """
        Perform comprehensive similarity analysis between two documents
        """
        # Process both documents
        primary_fragments = self.text_analyzer.fragment_and_normalize_content(primary_document)
        secondary_fragments = self.text_analyzer.fragment_and_normalize_content(secondary_document)
        
        # Calculate similarity metrics
        similarity_score, detailed_matches = self.text_analyzer.assess_content_overlap(
            primary_fragments, secondary_fragments
        )
        
        return {
            'primary_fragments': primary_fragments,
            'secondary_fragments': secondary_fragments,
            'similarity_score': similarity_score,
            'detailed_analysis': detailed_matches
        }
    
    def display_analysis_results(self, analysis_results):
        """
        Present the analysis results in a formatted manner
        """
        print("Primary Document Fragments:", analysis_results['primary_fragments'])
        print("Secondary Document Fragments:", analysis_results['secondary_fragments'])
        print(f"\nOverall Similarity Index: {analysis_results['similarity_score']:.2f}\n")
        
        print("Detailed Fragment Analysis:")
        print("-" * 50)
        
        for idx, match_data in enumerate(analysis_results['detailed_analysis'], 1):
            print(f"Fragment {idx}:")
            print(f"  Original: '{match_data['original']}'")
            print(f"  Best Match: '{match_data['matched']}'")
            print(f"  Confidence Level: {match_data['confidence']:.2f}")
            print()

def main():
    """
    Main execution function demonstrating plagiarism detection capabilities
    """
    # Sample documents for analysis
    reference_document = "The quick brown fox jumps over the lazy dog. It was a sunny day."
    suspect_document = "A speedy brown fox leaps over the sleepy dog. The day was bright and sunny."
    
    # Initialize detection system
    plagiarism_detector = PlagiarismDetectionSystem()
    
    # Perform analysis
    results = plagiarism_detector.execute_similarity_analysis(reference_document, suspect_document)
    
    # Display results
    plagiarism_detector.display_analysis_results(results)

if __name__ == "__main__":
    main()
