import unittest

class TestBasico(unittest.TestCase):
    def test_verificar_matematica(self):
        # Teste bobo apenas para validar o pipeline CI
        a = 10
        b = 20
        self.assertEqual(a + b, 30)

if __name__ == '__main__':
    unittest.main()
