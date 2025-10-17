import { Input } from "@/components/ui/input";
export default function Home() {
  return (
    <div className="justify-center items-left flex-col flex py-4 gap-4 px-4">
      <div className="text-5xl font-semibold">NFL Playoff Predictor</div>
      <div className="gap-2 flex flex-col">
        <div className="">NFL Team</div>
        <Input className="w-[20rem]"/>
        <div className="">Stat 1</div>
        <Input className="w-[20rem]"/>
      </div>
      <button className="bg-black text-white py-2 px-4 rounded-full font-bold w-[10rem]">Predict</button>
    </div>
  );
}
