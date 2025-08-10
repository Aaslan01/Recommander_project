#!/usr/bin/env python3
"""
Test script for the Recommander Project
This script tests that all components can be imported and basic functionality works.
"""

import sys
import os
import traceback

def test_imports():
    """Test that all project modules can be imported."""
    print("🧪 Testing imports...")
    
    modules_to_test = [
        "data_preprocessing",
        "cf_matrix_factorization", 
        "hybrid_recommender",
        "sentiment_analysis",
        "run_experiments",
        "evaluation",
        "topk_evaluation"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"  ⚠️  {module}: {e}")
            failed_imports.append(module)
    
    return failed_imports

def test_data_files():
    """Test that required data files exist."""
    print("\n📁 Testing data files...")
    
    required_files = [
        "data/u.data",
        "data/u.item", 
        "data/u.genre"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    return missing_files

def test_sentiment_generation():
    """Test sentiment generation functionality."""
    print("\n📊 Testing sentiment generation...")
    
    try:
        # Check if sentiment file exists, if not generate it
        sentiment_path = "data/sentiment_scores.csv"
        if not os.path.exists(sentiment_path):
            print("  📝 Generating sentiment data...")
            os.system("python sentiment_analysis.py")
            
        if os.path.exists(sentiment_path):
            print("  ✅ Sentiment data available")
            return True
        else:
            print("  ❌ Failed to generate sentiment data")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing sentiment: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test data preprocessing
        from data_preprocessing import load_data
        print("  ✅ Data preprocessing import successful")
        
        # Test matrix factorization
        from cf_matrix_factorization import MatrixFactorization
        print("  ✅ Matrix factorization import successful")
        
        # Test hybrid recommender
        from hybrid_recommender import HybridRecommender
        print("  ✅ Hybrid recommender import successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing functionality: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🎬 Recommander Project - System Test")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test data files
    missing_files = test_data_files()
    
    # Test sentiment generation
    sentiment_ok = test_sentiment_generation()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    if failed_imports:
        print(f"❌ Failed imports: {len(failed_imports)}")
        for module in failed_imports:
            print(f"   - {module}")
    else:
        print("✅ All imports successful")
    
    if missing_files:
        print(f"❌ Missing files: {len(missing_files)}")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("✅ All data files present")
    
    if sentiment_ok:
        print("✅ Sentiment data available")
    else:
        print("❌ Sentiment data issues")
    
    if functionality_ok:
        print("✅ Basic functionality working")
    else:
        print("❌ Basic functionality issues")
    
    # Overall status
    if not failed_imports and not missing_files and sentiment_ok and functionality_ok:
        print("\n🎉 All tests passed! Project is ready to run.")
        print("\n🚀 You can now run:")
        print("   python final_pipeline.py    # Complete pipeline")
        print("   python main.py              # Main interface")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("\n💡 Common solutions:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check data files are in the data/ directory")
        print("   - Ensure Python 3.8+ is being used")

if __name__ == "__main__":
    main() 