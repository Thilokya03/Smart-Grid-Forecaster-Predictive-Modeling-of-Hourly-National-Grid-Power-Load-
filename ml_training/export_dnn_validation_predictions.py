"""Export fold-matched DNN/LSTM validation artifacts.

This entry point is kept for the dashboard/report workflow. The actual
implementation lives in dnn_4fold_cv.py so there is one DNN validation
protocol to maintain.
"""

from dnn_4fold_cv import main


if __name__ == "__main__":
    main()
