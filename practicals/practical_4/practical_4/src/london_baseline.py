# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

def evaluate_places_london_baseline(filepath):
  """ Computes percent of correctly predicted birth places.

  Arguments:
    filepath: path to a file with our name, birth place data.

  Returns:
    (total, correct), floats
  """
  with open(filepath) as fin:
    lines = [x.strip().split('\t') for x in fin]
    if len(lines[0]) == 1:
      print('No gold birth places provided; returning (0,0)')
      return (0,0)
    true_places = [x[1] for x in lines]
    total = len(true_places)
    predicted_places = total * ['London']
    assert total == len(predicted_places)
    correct = len(list(filter(lambda x: x[0] == x[1],
      zip(true_places, predicted_places))))
    return (float(total),float(correct))
