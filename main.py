from focaccia.filters.blur import focacciaBlur
from focaccia.filters.ellipse import focacciaEllipse
from focaccia.filters.hsv import focacciaHsv
from focaccia.filters.radicalize import focacciaRadicalize

# foc1 = focacciaHsv(
#   path = 'assets/vegitables.jpg',
#   target_point = [250, 300],
#   lambda_ = 1e-50,
#   inside = True
# )
# foc1.switch(True, True, False)
# foc1.optimize(100)

# foc2 = focacciaBlur(
#   path = 'assets/vegitables.jpg',
#   target_point = [250, 300],
#   lambda_ = 1e-20,
#   inside = False
# )
# foc2.optimize(100)

# foc3 = focacciaEllipse(
#   path = 'assets/vegitables.jpg',
#   target_point = [300, 250],
#   lambda_ = 1e-5,
#   inside = False
# )
# foc3.optimize(100)

# foc4 = focacciaHsv(
#   path = 'assets/apples.jpeg',
#   target_point = [250, 300],
#   lambda_ = 1e-50,
#   inside = True
# )
# foc4.switch(True, True, False)
# foc4.optimize(100)

foc5 = focacciaRadicalize(
    path="assets/vegitables.jpg", target_point=[250, 300], lambda_=1e-20, inside=True
)
foc5.optimize(100)

# foc1.result_show()
# foc2.result_show()
# foc3.result_show()
# foc4.result_show()
foc5.result_show()
