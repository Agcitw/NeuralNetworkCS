using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace Viewer123
{
	public partial class Form1 : Form
	{
		private static int _iter;
		
		public Form1()
		{
			InitializeComponent();
		}

		private void Form1_Mousedown(object sender, MouseEventArgs e)
		{
			if (_iter + 1 > 10_000)
			{
				MessageBox.Show("Не надо");
				return;
			}
			_iter++;
			var rows = File.
				ReadAllLines(@"C:\Users\anton\RiderProjects\NeuralNetwork\NeuralNetwork\bin\Debug\netcoreapp3.1\mnist_test.csv").
				Take(_iter).ToList();
			var bmp = new Bitmap(28, 28);
			
			foreach (var values in rows.Select(row => row.Split(',')))
			{
				for (var i = 0; i < 28; i++)
				for (var j = 0; j < 28; j++)
					bmp.SetPixel(i, j, Convert.ToInt32(values[i + j * 28 + 1]) > 0 ? Color.Black : Color.White);
			}
			pictureBox1.Image = bmp;
			pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
			label1.Text = _iter.ToString();
		}

		private void button2_MouseDown(object sender, MouseEventArgs e)
		{
			if (_iter - 1 < 0)
			{
				MessageBox.Show("Не надо");
				return;
			}

			_iter--;
			var rows = File.
				ReadAllLines(@"C:\Users\anton\RiderProjects\NeuralNetwork\NeuralNetwork\bin\Debug\netcoreapp3.1\mnist_test.csv").
				Take(_iter).ToList();
			var bmp = new Bitmap(28, 28);
			
			foreach (var values in rows.Select(row => row.Split(',')))
			{
				for (var i = 0; i < 28; i++)
				for (var j = 0; j < 28; j++)
					bmp.SetPixel(i, j, Convert.ToInt32(values[i + j * 28 + 1]) > 0 ? Color.Black : Color.White);
			}
			pictureBox1.Image = bmp;
			pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
			label1.Text = _iter.ToString();
		}
	}
}