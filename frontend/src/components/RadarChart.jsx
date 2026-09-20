import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'

import { Radar } from 'react-chartjs-2'

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
)


function RadarChart({
  consistency = 0,
  entropy = 0,
  uncertainty = 0,
  confidence = 0,
}) {

  const data = {
    labels: [
      'Self-Consistency',
      'Semantic Entropy',
      'Uncertainty',
      'Confidence',
    ],

    datasets: [
      {
        label: 'Member 4 Analysis',

        data: [
          consistency,
          entropy,
          uncertainty,
          confidence,
        ],

        borderWidth: 2,

        pointRadius: 5,

        pointHoverRadius: 7,
      },
    ],
  }


  const options = {
    responsive: true,

    maintainAspectRatio: false,

    scales: {
      r: {
        min: 0,
        max: 1,

        ticks: {
          stepSize: 0.2,

          backdropColor: 'transparent',

          color: '#94a3b8',

          font: {
            size: 12,
          },
        },

        pointLabels: {
          color: '#cbd5e1',

          font: {
            size: 14,
            weight: '600',
          },
        },

        grid: {
          color: '#334155',
        },

        angleLines: {
          color: '#334155',
        },
      },
    },

    plugins: {
      legend: {
        position: 'top',

        labels: {
          color: '#cbd5e1',

          font: {
            size: 14,
          },
        },
      },

      tooltip: {
        callbacks: {
          label: function (context) {
            return `${context.dataset.label}: ${context.raw.toFixed(4)}`
          },
        },
      },
    },
  }


  return (
    <div
      style={{
        width: '100%',
        height: '500px',
      }}
    >
      <Radar
        data={data}
        options={options}
      />
    </div>
  )
}


export default RadarChart